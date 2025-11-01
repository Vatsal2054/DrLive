import axios from 'axios';
// Environment variables for secure API key storage
const GOOGLE_MAPS_API_KEY = process.env.GOOGLE_MAPS_API_KEY;

/**
 * Get nearby pharmacies based on location
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
const getNearbyPharmacies = async (req, res) => {
  try {
    // Get location parameters from query or use defaults
    const { lat = 22.3072, lng = 73.1812, radius = 25000 } = req.query;
    
    // Validate input parameters
    if (isNaN(parseFloat(lat)) || isNaN(parseFloat(lng)) || isNaN(parseInt(radius))) {
      return res.status(400).json({ 
        success: false, 
        message: 'Invalid parameters. Latitude, longitude, and radius must be numbers' 
      });
    }

    // Make request to Google Places API
    const response = await axios.get('https://maps.googleapis.com/maps/api/place/nearbysearch/json', {
      params: {
        location: `${lat},${lng}`,
        radius: radius,
        type: 'pharmacy',
        key: GOOGLE_MAPS_API_KEY
      }
    });

    // Sanitize and filter the response data for security
    const sanitizedResults = response.data.results.map(pharmacy => ({
      id: pharmacy.place_id,
      name: pharmacy.name,
      place_id: pharmacy.place_id,
      business_status: pharmacy.business_status,
      rating: pharmacy.rating || 0,
      user_ratings_total: pharmacy.user_ratings_total || 0,
      vicinity: pharmacy.vicinity,
      types: pharmacy.types || [],
      opening_hours: pharmacy.opening_hours || { open_now: false },
      location: pharmacy.geometry?.location || { lat: 0, lng: 0 }
    }));

    // Return success with sanitized data
    return res.status(200).json({
      success: true,
      results: sanitizedResults,
      next_page_token: response.data.next_page_token || null
    });
  } catch (error) {
    console.error('Error fetching pharmacy data:', error);
    return res.status(500).json({ 
      success: false, 
      message: 'Error fetching pharmacy data',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

// Get details for a specific pharmacy
const getPharmacyDetails = async (req, res) => {
  try {
    const { placeId } = req.params;
    
    if (!placeId) {
      return res.status(400).json({ 
        success: false, 
        message: 'Place ID is required' 
      });
    }

    const response = await axios.get('https://maps.googleapis.com/maps/api/place/details/json', {
      params: {
        place_id: placeId,
        fields: 'name,formatted_address,formatted_phone_number,opening_hours,website,rating,review,geometry',
        key: GOOGLE_MAPS_API_KEY
      }
    });

    if (!response.data.result) {
      return res.status(404).json({
        success: false,
        message: 'Pharmacy not found'
      });
    }

    return res.status(200).json({
      success: true,
      result: response.data.result
    });
  } catch (error) {
    console.error('Error fetching pharmacy details:', error);
    return res.status(500).json({ 
      success: false, 
      message: 'Error fetching pharmacy details',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};


export { getNearbyPharmacies, getPharmacyDetails };