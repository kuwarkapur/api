To use this REST API, you can send requests to the following endpoints:

1. GET /predict/<path:url>: Send a GET request to this endpoint with the URL of the image as a path parameter to predict the food item in the image. Example: http://localhost:5000/predict/https://example.com/image.jpg

2. POST /predictbase64: Send a POST request to this endpoint with the image URL in the request body as a JSON object. Example: {"url": "https://localhost:5000/predictbase64/base64ofimage"}

3. GET /pred/<string:name>: Send a GET request to this endpoint with the name of the food item as a path parameter to retrieve its details from the database. Example: http://localhost:5000/pred/kale

