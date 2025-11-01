from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import PyPDF2
import os
import json
import re
import logging
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
from pymongo import MongoClient
from bson import ObjectId
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

app = Flask(__name__)
# Improved CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:3000","*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})
app.json_encoder = MongoJSONEncoder

# MongoDB setup
db = None
doctors_collection = None
users_collection = None

if MONGO_URI:
    try:
        db_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=15000,  # Increased timeout for DNS resolution
            connectTimeoutMS=10000,
            socketTimeoutMS=12000,
            retryWrites=True,
            tls=True,
            tlsAllowInvalidCertificates=True
        )
        # Test connection with a longer timeout
        db_client.server_info()
        logger.info("MongoDB Atlas connection successful!")
        
        # Extract database name from URI
        try:
            parsed_uri = urlparse(MONGO_URI)
            db_name = parsed_uri.path.lstrip('/').split('?')[0] if parsed_uri.path else None
            if not db_name or db_name.startswith('@'):
                # Try to extract from connection string directly
                if '/' in MONGO_URI and '?' in MONGO_URI:
                    db_name = MONGO_URI.split('/')[-1].split('?')[0]
                elif '/' in MONGO_URI:
                    db_name = MONGO_URI.split('/')[-1]
                else:
                    db_name = "test"  # Default fallback
            if not db_name or db_name == '':
                db_name = "test"
            logger.info(f"Using database: {db_name}")
            db = db_client[db_name]
        except Exception as db_name_error:
            logger.warning(f"Could not extract database name: {db_name_error}, using default 'test'")
            db = db_client["test"]
        
        doctors_collection = db["doctors"]
        users_collection = db["users"]
        logger.info("MongoDB collections initialized successfully")
    except Exception as e:
        logger.error(f"MongoDB connection error: {str(e)}")
        db = None
        doctors_collection = None
        users_collection = None
        logger.warning("Running with database functionality disabled")
else:
    logger.warning("MONGO_URI not set in environment variables, running with database functionality disabled")

# Initialize Gemini models
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # List available models to identify the correct model name
    models = genai.list_models()
    model_names = [model.name for model in models]
    logger.info(f"Available models count: {len(model_names)}")
    gemini_model_name = None
    
    # Priority order for model selection (fastest first for better user experience)
    preferred_models = [
        "gemini-2.5-flash",         # Fastest: Latest stable flash model (for speed)
        "gemini-2.5-pro",           # Best: Latest stable pro model (fallback if flash not available)
        "gemini-2.0-flash",         # Fast: Stable flash model
        "gemini-2.0-pro",           # Good: Stable pro model
    ]
    
    # Try to find preferred models first (skip preview/experimental versions)
    for preferred in preferred_models:
        for model_name in model_names:
            clean_name = model_name.replace("models/", "").lower()
            # Match exact model name, skip preview/exp versions
            if preferred.lower() in clean_name and "preview" not in clean_name and "exp" not in clean_name:
                gemini_model_name = model_name
                logger.info(f"Selected preferred Gemini model: {gemini_model_name}")
                break
        if gemini_model_name:
            break
    
    # If no preferred stable model found, try preview versions
    if not gemini_model_name:
        preview_preferred = [
            "gemini-2.5-pro-preview",
            "gemini-2.5-flash-preview",
            "gemini-2.0-pro-exp",
        ]
        for preferred in preview_preferred:
            for model_name in model_names:
                clean_name = model_name.replace("models/", "").lower()
                if preferred.lower() in clean_name:
                    gemini_model_name = model_name
                    logger.info(f"Selected Gemini preview model: {gemini_model_name}")
                    break
            if gemini_model_name:
                break
    
    # If still not found, look for any pro model (excluding embeddings)
    if not gemini_model_name:
        for model_name in model_names:
            clean_name = model_name.replace("models/", "").lower()
            if "gemini" in clean_name and "pro" in clean_name and "embedding" not in clean_name:
                gemini_model_name = model_name
                logger.info(f"Selected Gemini pro model: {gemini_model_name}")
                break
    
    # If still not found, use any gemini model (excluding embeddings, imagen, veo)
    if not gemini_model_name:
        for model_name in model_names:
            clean_name = model_name.replace("models/", "").lower()
            if ("gemini" in clean_name and "embedding" not in clean_name and 
                "imagen" not in clean_name and "veo" not in clean_name and
                "gemma" not in clean_name):
                gemini_model_name = model_name
                logger.info(f"Selected Gemini model: {gemini_model_name}")
                break
    
    if not gemini_model_name:
        logger.error("No suitable Gemini model found in available models!")
        gemini_model_name = None
    else:
        logger.info(f"Successfully initialized with model: {gemini_model_name}")
    
except Exception as e:
    logger.error(f"Gemini API initialization error: {str(e)}")
    gemini_model_name = None

class MedicalSystem:
    SPECIALIZATIONS = {
        "Cardiologist": {
            "keywords": ["heart", "chest", "blood pressure", "lungs"],
            "description": "Heart and cardiovascular system specialist"
        },
        "Dermatologist": {
            "keywords": ["skin", "acne", "rash"],
            "description": "Skin, hair, and nail conditions specialist"
        },
        "Pediatrician": {
            "keywords": ["child", "infant", "pediatric"],
            "description": "Children's health specialist"
        },
        "Neurologist": {
            "keywords": ["brain", "headache", "nerve"],
            "description": "Brain, spinal cord, and nervous system specialist"
        },
        "Orthopaedic": {
            "keywords": ["bone", "joint", "muscle", "fracture", "broken", "sprain", "strain"],
            "description": "Bone and joint specialist"
        },
        "Psychiatrist": {
            "keywords": ["anxiety", "depression", "mental"],
            "description": "Mental health specialist"
        },
        "General Medicine": {
            "keywords": ["fever", "cold", "cough"],
            "description": "Primary care and general health conditions"
        }
    }

    def __init__(self):
        try:
            global gemini_model_name
            if gemini_model_name:
                self.model = genai.GenerativeModel(gemini_model_name)
                logger.info(f"Medical system initialized with model: {gemini_model_name}")
            else:
                logger.error("No valid Gemini model name available")
                self.model = None
        except Exception as e:
            logger.error(f"Gemini model initialization error: {str(e)}")
            self.model = None

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from PDF file."""
        text = []
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            return ' '.join(text)
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    @staticmethod
    def clean_json_response(response_text: str) -> Dict[str, Any]:
        """Clean and parse JSON response."""
        # Remove markdown code blocks if present
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from text using a more robust regex
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            match = re.search(json_pattern, response_text, re.DOTALL)
            if match:
                try:
                    json_str = match.group(0)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            # Try to find and extract the largest JSON-like structure
            try:
                matches = re.finditer(r'\{.*?\}', response_text, re.DOTALL)
                for match in reversed(list(matches)):  # Try largest first
                    try:
                        return json.loads(match.group(0))
                    except json.JSONDecodeError:
                        continue
            except Exception:
                pass
                
        logger.warning(f"Failed to parse JSON response. Raw response: {response_text[:200]}")
        return {"error": "Failed to parse AI response", "raw_response": response_text[:500]}

    def get_precautions_and_recommendations(self, symptoms: str, language: str = "en-US") -> Dict[str, Any]:
        """Generate precautions and recommendations based on symptoms using Gemini."""
        if self.model is None:
            logger.error("Model is not initialized, returning fallback response")
            return {
                "error": "AI model not initialized",
                "initial_assessment": {
                    "severity": "unknown",
                    "immediate_action_required": False,
                    "seek_emergency": False
                },
                "precautions": [
                    {
                        "category": "General Advice",
                        "measures": ["Please consult with a healthcare professional for proper evaluation"],
                        "priority": "high"
                    }
                ]
            }
            
        # Determine language for the response
        language_prompt = ""
        if language != "en-US":
            language_prompt = f"Respond in {language} language. "
            
        prompt = f"""{language_prompt}Analyze symptoms: {symptoms}

Provide concise medical recommendations in JSON format. IMPORTANT: Include practical home remedies that are safe and appropriate for these symptoms.

{{
    "initial_assessment": {{
        "severity": "mild/moderate/severe",
        "immediate_action_required": false,
        "seek_emergency": false,
        "assessment_note": "Brief assessment"
    }},
    "precautions": [
        {{"category": "Action", "measures": ["Measure 1", "Measure 2"], "priority": "high"}}
    ],
    "lifestyle_recommendations": [
        {{"area": "Care", "suggestions": ["Suggestion 1"], "duration": "temporary"}}
    ],
    "home_remedies": [
        {{"remedy": "Remedy name", "instructions": "How to use it", "caution": "Any warnings"}},
        {{"remedy": "Another remedy", "instructions": "Steps to follow", "caution": "Safety notes"}}
    ],
    "when_to_seek_emergency": ["Emergency sign 1", "Emergency sign 2"],
    "general_advice": "Brief specific advice"
}}

IMPORTANT: Always provide at least 2-3 practical, safe home remedies relevant to the symptoms. Do not leave home_remedies empty unless symptoms are severe and require immediate medical attention only.

Return ONLY valid JSON."""

        try:
            logger.info(f"Sending prompt to Gemini for symptoms: {symptoms[:100]}...")
            
            # Use GenerationConfig for better control - optimized for speed
            def call_gemini():
                try:
                    from google.generativeai.types import GenerationConfig
                    generation_config = GenerationConfig(
                        temperature=0.3,  # Balanced for quality and speed
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=1500,  # Optimized for faster responses
                    )
                    return self.model.generate_content(prompt, generation_config=generation_config)
                except (ImportError, AttributeError) as config_error:
                    logger.warning(f"GenerationConfig not available: {config_error}, using default")
                    return self.model.generate_content(prompt)
            
            # Call Gemini with a timeout to prevent hanging
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(call_gemini)
                    response = future.result(timeout=14)  # 14 second timeout (less than client's 15s)
            except FuturesTimeoutError:
                logger.warning("Gemini API call timed out after 14 seconds - returning fallback response")
                # Return timeout response directly instead of raising exception
                return {
                    "initial_assessment": {
                        "severity": "moderate",
                        "immediate_action_required": False,
                        "seek_emergency": False,
                        "assessment_note": "Based on your symptoms, a medical consultation is recommended. The AI analysis timed out, but you should still consult with a healthcare provider."
                    },
                    "precautions": [
                        {
                            "category": "Medical Consultation",
                            "measures": [
                                "Schedule an appointment with a healthcare provider as soon as possible",
                                "Be prepared to describe your symptoms in detail",
                                "Bring a list of any medications you're currently taking",
                                "Note when symptoms started and any patterns you've noticed"
                            ],
                            "priority": "high"
                        },
                        {
                            "category": "Symptom Monitoring",
                            "measures": [
                                "Keep track of when symptoms occur",
                                "Note any triggers or patterns",
                                "Monitor symptom intensity and duration",
                                "Record any changes in symptoms"
                            ],
                            "priority": "medium"
                        }
                    ],
                    "lifestyle_recommendations": [
                        {
                            "area": "General Care",
                            "suggestions": [
                                "Get adequate rest and sleep (7-9 hours)",
                                "Stay well-hydrated throughout the day",
                                "Avoid activities that worsen symptoms",
                                "Maintain a balanced diet"
                            ],
                            "duration": "ongoing"
                        }
                    ],
                    "home_remedies": [],
                    "when_to_seek_emergency": [
                        "If symptoms worsen significantly or rapidly",
                        "If new severe symptoms develop",
                        "If you experience difficulty breathing",
                        "If pain becomes severe or unmanageable",
                        "If symptoms persist or worsen after 24-48 hours"
                    ],
                    "general_advice": "While the AI analysis timed out, consulting with the recommended specialist is important for proper diagnosis and treatment planning."
                }
            except Exception as api_error:
                logger.error(f"Gemini API call error: {str(api_error)}")
                raise
            
            if not response:
                raise Exception("Empty response from Gemini API")
            
            # Check for blocked content or safety filters
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    # Finish reason 2 = SAFETY (blocked by safety filters)
                    # Finish reason 3 = RECITATION (blocked due to recitation)
                    # Finish reason 4 = OTHER
                    if finish_reason in [2, 3, 4]:
                        logger.warning(f"Response blocked by safety filter, finish_reason: {finish_reason}")
                        raise Exception(f"Response was blocked (finish_reason: {finish_reason}). Try rephrasing your symptoms.")
            
            # Safely get response text
            try:
                response_text = response.text
            except ValueError as ve:
                # Response doesn't have valid text (blocked or filtered)
                logger.error(f"Cannot access response.text: {ve}")
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        raise Exception(f"Response blocked. Finish reason: {candidate.finish_reason}")
                raise Exception("Unable to get response text - content may have been filtered")
            
            if not response_text or len(response_text.strip()) == 0:
                raise Exception("Empty response text from Gemini")
                
            logger.info(f"Successfully received response from Gemini. Length: {len(response_text)}")
            logger.debug(f"Response preview: {response_text[:300]}")
            
            result = self.clean_json_response(response_text)
            
            # Check if parsing failed
            if "error" in result and "Failed to parse" in result.get("error", ""):
                logger.error(f"JSON parsing failed. Raw response: {result.get('raw_response', '')[:500]}")
                # Try to retry once with a simpler prompt if parsing fails
                raise Exception("JSON parsing failed - invalid response format")
            
            # Ensure all required fields exist and are not empty
            if "initial_assessment" not in result or not result.get("initial_assessment"):
                result["initial_assessment"] = {
                    "severity": "moderate",
                    "immediate_action_required": False,
                    "seek_emergency": False,
                    "assessment_note": "Based on the provided symptoms"
                }
            
            # If severity is "unknown", change it to a reasonable default
            assessment = result.get("initial_assessment", {})
            if assessment.get("severity") == "unknown" or not assessment.get("severity"):
                result["initial_assessment"]["severity"] = "moderate"
                if "assessment_note" not in assessment or not assessment.get("assessment_note"):
                    result["initial_assessment"]["assessment_note"] = "Moderate severity based on symptoms - requires medical consultation"
            
            # Ensure arrays exist and are not empty
            if "precautions" not in result or not result.get("precautions") or len(result.get("precautions", [])) == 0:
                result["precautions"] = [
                    {
                        "category": "Medical Consultation",
                        "measures": ["Schedule an appointment with a healthcare provider for proper diagnosis"],
                        "priority": "high"
                    }
                ]
            
            if "lifestyle_recommendations" not in result:
                result["lifestyle_recommendations"] = []
                
            # Ensure home_remedies exists and has content
            if "home_remedies" not in result:
                result["home_remedies"] = []
            
            # If home_remedies is empty but symptoms are not severe, provide some basic safe remedies
            if not result.get("home_remedies") or len(result.get("home_remedies", [])) == 0:
                assessment = result.get("initial_assessment", {})
                severity = assessment.get("severity", "moderate")
                if severity != "severe" and not assessment.get("seek_emergency", False):
                    # Provide basic safe home remedies based on symptoms
                    symptoms_lower = symptoms.lower()
                    basic_remedies = []
                    
                    if any(word in symptoms_lower for word in ["pain", "ache", "sore", "hurt"]):
                        basic_remedies.append({
                            "remedy": "Rest and Ice/Heat Therapy",
                            "instructions": "Rest the affected area. Apply ice pack (wrapped in cloth) for 15-20 minutes if acute pain, or warm compress for chronic pain. Repeat every 2-3 hours.",
                            "caution": "Do not apply ice directly to skin. Stop if pain worsens."
                        })
                    
                    if any(word in symptoms_lower for word in ["cough", "cold", "congestion", "sore throat"]):
                        basic_remedies.append({
                            "remedy": "Warm Salt Water Gargle",
                            "instructions": "Mix 1/2 teaspoon of salt in a glass of warm water. Gargle for 30 seconds, then spit. Repeat 2-3 times daily.",
                            "caution": "Do not swallow the salt water. Use lukewarm, not hot water."
                        })
                        basic_remedies.append({
                            "remedy": "Honey and Warm Water",
                            "instructions": "Mix 1-2 teaspoons of honey in a cup of warm water or herbal tea. Drink slowly 2-3 times daily.",
                            "caution": "Do not give honey to children under 1 year old."
                        })
                    
                    if any(word in symptoms_lower for word in ["fever", "temperature"]):
                        basic_remedies.append({
                            "remedy": "Stay Hydrated",
                            "instructions": "Drink plenty of fluids (water, electrolyte solutions, clear broths). Aim for 8-10 glasses daily.",
                            "caution": "Seek medical attention if fever persists beyond 3 days or exceeds 103°F (39.4°C)."
                        })
                        basic_remedies.append({
                            "remedy": "Cool Compress",
                            "instructions": "Apply a cool, damp cloth to forehead, wrists, or back of neck. Replace when warm.",
                            "caution": "Do not use ice-cold water. Monitor temperature regularly."
                        })
                    
                    if any(word in symptoms_lower for word in ["stomach", "nausea", "indigestion", "upset"]):
                        basic_remedies.append({
                            "remedy": "Ginger Tea",
                            "instructions": "Steep fresh ginger slices in hot water for 5-10 minutes. Drink slowly, 1-2 cups daily.",
                            "caution": "Avoid if you have gallstones or are taking blood thinners."
                        })
                        basic_remedies.append({
                            "remedy": "BRAT Diet",
                            "instructions": "If experiencing digestive issues, consume bland foods: Bananas, Rice, Applesauce, Toast. Eat small, frequent meals.",
                            "caution": "Seek medical attention if symptoms persist or worsen."
                        })
                    
                    if not basic_remedies:
                        # General safe remedies for any mild symptoms
                        basic_remedies.append({
                            "remedy": "Adequate Rest",
                            "instructions": "Get 7-9 hours of sleep per night. Take short breaks during the day if needed.",
                            "caution": "Ensure proper sleep hygiene and a comfortable environment."
                        })
                        basic_remedies.append({
                            "remedy": "Stay Hydrated",
                            "instructions": "Drink 8-10 glasses of water daily. Include warm beverages if comforting.",
                            "caution": "Avoid excessive caffeine or alcohol which can worsen symptoms."
                        })
                    
                    if basic_remedies:
                        result["home_remedies"] = basic_remedies[:3]  # Limit to 3 remedies
                        logger.info(f"Added {len(basic_remedies)} home remedies based on symptom analysis")
                
            if "when_to_seek_emergency" not in result or not result.get("when_to_seek_emergency"):
                result["when_to_seek_emergency"] = [
                    "If symptoms worsen significantly",
                    "If new severe symptoms develop",
                    "If symptoms persist beyond 3-5 days"
                ]
            
            logger.info(f"Successfully generated recommendations for symptoms")
            return result
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"Precautions generation error: {error_str}", exc_info=True)
            
            # Check if it's a safety filter issue
            if "blocked" in error_str.lower() or "finish_reason" in error_str.lower():
                # Response was blocked - provide helpful medical advice without AI
                return {
                    "initial_assessment": {
                        "severity": "moderate",
                        "immediate_action_required": False,
                        "seek_emergency": False,
                        "assessment_note": "Based on your symptoms, a medical consultation is recommended for proper evaluation."
                    },
                    "precautions": [
                        {
                            "category": "Medical Consultation",
                            "measures": [
                                "Schedule an appointment with a healthcare provider",
                                "Be prepared to describe your symptoms in detail",
                                "Bring a list of any medications you're currently taking"
                            ],
                            "priority": "high"
                        },
                        {
                            "category": "Symptom Monitoring",
                            "measures": [
                                "Keep track of when symptoms occur",
                                "Note any triggers or patterns",
                                "Monitor symptom intensity and duration"
                            ],
                            "priority": "medium"
                        }
                    ],
                    "lifestyle_recommendations": [
                        {
                            "area": "General Care",
                            "suggestions": [
                                "Get adequate rest and sleep",
                                "Stay well-hydrated",
                                "Avoid activities that worsen symptoms"
                            ],
                            "duration": "temporary"
                        }
                    ],
                    "home_remedies": [],
                    "when_to_seek_emergency": [
                        "If symptoms worsen significantly",
                        "If new severe symptoms develop",
                        "If you experience difficulty breathing",
                        "If pain becomes severe or unmanageable"
                    ],
                    "general_advice": "For your symptoms, consulting with the recommended specialist will help determine the best course of treatment."
                }
            
            # For other errors, return helpful medical advice
            return {
                "initial_assessment": {
                    "severity": "moderate",
                    "immediate_action_required": False,
                    "seek_emergency": False,
                    "assessment_note": "Based on your symptoms, medical consultation is recommended for proper evaluation and treatment."
                },
                "precautions": [
                    {
                        "category": "Medical Consultation",
                        "measures": [
                            "Schedule an appointment with a healthcare provider",
                            "Describe your symptoms clearly and in detail",
                            "Mention how long you've been experiencing these symptoms"
                        ],
                        "priority": "high"
                    },
                    {
                        "category": "Self-Care",
                        "measures": [
                            "Get adequate rest",
                            "Stay hydrated",
                            "Monitor your symptoms"
                        ],
                        "priority": "medium"
                    }
                ],
                "lifestyle_recommendations": [
                    {
                        "area": "Rest and Recovery",
                        "suggestions": [
                            "Ensure sufficient sleep (7-9 hours)",
                            "Avoid strenuous activities",
                            "Take breaks when needed"
                        ],
                        "duration": "temporary"
                    }
                ],
                "home_remedies": [],
                "when_to_seek_emergency": [
                    "If symptoms worsen significantly",
                    "If new severe symptoms appear",
                    "If you have difficulty breathing",
                    "If pain becomes severe"
                ],
                "general_advice": "Consult with the recommended specialist for proper medical evaluation and personalized treatment plan."
            }

    @staticmethod
    def find_best_specialty(symptoms: str) -> str:
        """Find best matching specialty based on symptoms."""
        symptoms_lower = symptoms.lower()
        for specialty, info in MedicalSystem.SPECIALIZATIONS.items():
            for keyword in info["keywords"]:
                if keyword in symptoms_lower:
                    return specialty
        return "General Medicine"

    @staticmethod
    def get_doctors_for_specialty(specialty: str) -> list:
        """Get available doctors for a specialty."""
        if doctors_collection is None or users_collection is None:
            # Return empty list if database is not available
            logger.warning("Database connection not available, returning empty doctors list")
            return []
            
        try:
            logger.info(f"Searching for doctors with specialty: '{specialty}'")
            
            # Try multiple query variations to handle case sensitivity
            query_variations = [
                {"specialization": specialty, "isAvailable": True},
                {"specialization": specialty, "isAvailable": "true"},
                {"specialization": specialty},  # Without availability check
                {"specialization": {"$regex": f"^{re.escape(specialty)}$", "$options": "i"}, "isAvailable": True},  # Case insensitive
                {"specialization": {"$regex": f"^{re.escape(specialty)}$", "$options": "i"}},
            ]
            
            doctors = []
            doctor_ids_found = set()  # To avoid duplicates
            
            for query in query_variations:
                if doctors:
                    break  # Stop if we found doctors
                    
                try:
                    doctor_cursor = doctors_collection.find(
                        query,
                        {"userId": 1, "degree": 1, "experience": 1, "workingPlace": 1, "specialization": 1}
                    ).limit(10)
                    
                    count = doctors_collection.count_documents(query)
                    logger.info(f"Query {query} found {count} doctors")
                    
                    for doctor in doctor_cursor:
                        doctor_id_raw = doctor.get("userId")
                        if not doctor_id_raw:
                            continue
                        
                        # Convert to string for comparison
                        doctor_id_str = str(doctor_id_raw)
                        if doctor_id_str in doctor_ids_found:
                            continue
                        
                        # Convert userId to ObjectId if it's a string
                        if isinstance(doctor_id_raw, str):
                            try:
                                doctor_id = ObjectId(doctor_id_raw)
                            except Exception:
                                logger.warning(f"Invalid userId format: {doctor_id_raw}")
                                continue
                        elif isinstance(doctor_id_raw, ObjectId):
                            doctor_id = doctor_id_raw
                        else:
                            continue
                        
                        # Try to find user with this ID
                        user = users_collection.find_one(
                            {"_id": doctor_id},
                            {"firstName": 1, "lastName": 1, "address": 1}
                        )
                        
                        if user:
                            doctor_ids_found.add(doctor_id_str)
                            
                            # Handle address structure - could be nested or flat
                            location = doctor.get("workingPlace", "")  # Use workingPlace from doctor document
                            if not location:
                                address = user.get("address", {})
                                if isinstance(address, dict):
                                    location = address.get("city", "") or address.get("workingPlace", "")
                                elif isinstance(address, str):
                                    location = address
                            
                            doctors.append({
                                "doctorId": str(doctor_id),
                                "name": f"{user.get('firstName', '')} {user.get('lastName', '')}".strip(),
                                "degree": doctor.get("degree", ""),
                                "experience": doctor.get("experience", ""),
                                "location": location
                            })
                            logger.info(f"Added doctor: {user.get('firstName', '')} {user.get('lastName', '')}")
                    if doctors:
                        break
                        
                except Exception as query_error:
                    logger.warning(f"Query failed for {query}: {str(query_error)}")
                    continue
            
            if not doctors:
                # Debug: Check what's actually in the database
                try:
                    all_doctors = list(doctors_collection.find({}).limit(3))
                    logger.info(f"Sample doctors in DB: {[{'spec': d.get('specialization'), 'available': d.get('isAvailable')} for d in all_doctors]}")
                except Exception:
                    pass
            
            logger.info(f"Returning {len(doctors)} doctors for specialty: {specialty}")
            return doctors
        except Exception as e:
            logger.error(f"Database error: {str(e)}", exc_info=True)
            return []

    def analyze_medical_report(self, text: str, language: str = "en-US") -> Dict[str, Any]:
        """Analyze medical report using Google's Gemini model."""
        if self.model is None:
            logger.error("Model is not initialized, returning fallback response")
            return {"error": "AI model not initialized"}
            
        # Determine language for the response
        language_prompt = ""
        if language != "en-US":
            language_prompt = f"Respond in {language} language. "
            
        prompt = f"""
        {language_prompt}Analyze this medical report as a specialized medical AI. Provide a detailed analysis in JSON format:

        Guidelines:
        1. Extract ALL symptoms mentioned, even mild ones
        2. List ALL possible diseases that match the symptoms
        3. Consider test results and vital signs if present
        4. Recommend specialists based on symptoms and possible conditions
        5. Provide a comprehensive summary of the findings
        6. Include severity assessment of the overall condition

        Required JSON structure:
        {{
            "summary": {{
                "overview": "Brief overview of the case how diagnostic it and need to be reviewed by a specialist",
                "severity_assessment": "mild/moderate/severe",
                "key_findings": ["list of important findings"],
                "urgent_attention": "yes/no",
                "follow_up_timeline": "immediate/within week/routine"
            }},
            "symptoms": [
                {{
                    "symptom": "detailed symptom",
                    "severity": "mild/moderate/severe",
                    "duration": "duration if mentioned",
                    "related_conditions": ["possible related conditions"]
                }}
            ],
            "possible_diseases": [
                {{
                    "disease": "disease name",
                    "confidence": "high/medium/low",
                    "reasoning": "brief explanation",
                    "common_complications": ["possible complications"]
                }}
            ],
            "recommended_doctor": {{
                "primary": {{
                    "specialist": "main specialist needed",
                    "specialty_area": "specific area of expertise",
                    "urgency": "immediate/soon/routine"
                }},
                "secondary": {{
                    "specialist": "additional specialist if needed",
                    "specialty_area": "specific area of expertise",
                    "urgency": "immediate/soon/routine"
                }},
                "reasoning": "explanation for specialist choices"
            }},
            "precautions": [
                {{
                    "precaution": "specific precaution",
                    "importance": "critical/important/recommended",
                    "duration": "how long to follow",
                    "details": "additional details"
                }}
            ],
            "additional_tests": [
                {{
                    "test": "test name",
                    "purpose": "why it's needed",
                    "urgency": "immediate/soon/routine"
                }}
            ],
            "lifestyle_recommendations": [
                {{
                    "category": "diet/exercise/sleep/etc",
                    "recommendation": "specific advice",
                    "importance": "high/medium/low"
                }}
            ]
        }}

        Medical Report:
        {text}

        Ensure the response is ONLY the JSON object with no additional text.
        """

        try:
            response = self.model.generate_content(prompt)
            result = self.clean_json_response(response.text)
            
            # Validate and provide defaults for missing fields
            default_response = {
                "summary": {
                    "overview": "Unable to generate summary due to insufficient information",
                    "severity_assessment": "unknown",
                    "key_findings": ["No significant findings detected"],
                    "urgent_attention": "unknown",
                    "follow_up_timeline": "routine"
                },
                "symptoms": [{"symptom": "No symptoms detected", "severity": "unknown", "duration": "unknown", "related_conditions": []}],
                "possible_diseases": [{"disease": "Unable to determine", "confidence": "low", "reasoning": "Insufficient information", "common_complications": []}],
                "recommended_doctor": {
                    "primary": {
                        "specialist": "General Medicine",
                        "specialty_area": "General health assessment",
                        "urgency": "routine"
                    },
                    "secondary": None,
                    "reasoning": "Default recommendation due to insufficient information"
                },
                "precautions": [{"precaution": "Consult a healthcare provider", "importance": "critical", "duration": "until medical consultation", "details": "Seek professional medical advice"}],
                "additional_tests": [{"test": "General health assessment", "purpose": "Baseline health evaluation", "urgency": "routine"}],
                "lifestyle_recommendations": [{"category": "general", "recommendation": "Maintain healthy lifestyle", "importance": "high"}]
            }
            
            # Merge with defaults for any missing fields
            for key in default_response:
                if key not in result or not result[key]:
                    result[key] = default_response[key]

            # Add specialization details
            if "recommended_doctor" in result:
                primary_specialist = result["recommended_doctor"]["primary"]["specialist"]
                if primary_specialist in self.SPECIALIZATIONS:
                    result["recommended_doctor"]["primary"]["specialty_description"] = self.SPECIALIZATIONS[primary_specialist]
                
                if result["recommended_doctor"]["secondary"] is not None:
                    secondary_specialist = result["recommended_doctor"]["secondary"]["specialist"]
                    if secondary_specialist in self.SPECIALIZATIONS:
                        result["recommended_doctor"]["secondary"]["specialty_description"] = self.SPECIALIZATIONS[secondary_specialist]
                    
            return result
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "raw_response": None
            }

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "message": "Service is running"}), 200

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Handle medical report analysis requests."""
    if request.method == 'OPTIONS':
        return '', 204
        
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid or no file selected"}), 400
    
    # Get language from request
    data = request.form.to_dict()
    language = data.get("language", "en-US")
    
    medical_system = MedicalSystem()
    pdf_path = f"temp_{file.filename}"
    
    try:
        file.save(pdf_path)
        text = medical_system.extract_text_from_pdf(pdf_path)
        result = medical_system.analyze_medical_report(text, language)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

@app.route("/recommend", methods=["POST", "OPTIONS"])
def recommend():
    """Handle symptom-based doctor recommendations with dynamic precautions."""
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.get_json()
        logger.info(f"Received request data: {data}")
        
        if not data:
            logger.warning("No JSON data received in request")
            return jsonify({"error": "No data provided in request"}), 400
            
        symptoms = data.get("symptoms", "")
        language = data.get("language", "en-US")
        
        if not symptoms:
            return jsonify({"error": "Symptoms are required."}), 400

        medical_system = MedicalSystem()
        
        # Get specialty and doctors (fast operations)
        specialty = medical_system.find_best_specialty(symptoms)
        doctors = medical_system.get_doctors_for_specialty(specialty)
        
        # Get dynamic precautions and recommendations (this may take time)
        try:
            precautions_data = medical_system.get_precautions_and_recommendations(symptoms, language)
        except Exception as e:
            logger.error(f"Error getting precautions: {str(e)}")
            # Return partial response with doctors if available
            precautions_data = {
                "error": "Unable to generate recommendations at this time",
                "initial_assessment": {
                    "severity": "moderate",
                    "immediate_action_required": False,
                    "seek_emergency": False,
                    "assessment_note": "Recommendations unavailable - please consult with available doctors"
                },
                "precautions": [
                    {
                        "category": "Immediate Action",
                        "measures": ["Schedule a consultation with the recommended specialist"],
                        "priority": "high"
                    }
                ],
                "lifestyle_recommendations": [],
                "home_remedies": [],
                "when_to_seek_emergency": []
            }
        
        response = {
            "recommended_specialty": specialty,
            "specialty_description": medical_system.SPECIALIZATIONS[specialty]["description"],
            "available_doctors": doctors,
            "doctors_available": len(doctors) > 0,  # Flag indicating whether doctors are available
            "precautions_and_recommendations": precautions_data
        }
        
        # Add a message when no doctors are available
        if not doctors:
            response["doctor_availability_message"] = f"No {specialty} doctors are currently available."
        
        return jsonify(response)
            
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)