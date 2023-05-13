To use this REST API, you can send requests to the following endpoints:

1. GET /predict/<path:url>: Send a GET request to this endpoint with the URL of the image as a path parameter to predict the food item in the image. Example: http://localhost:5000/predict/https://example.com/image.jpg

2. POST /predict: Send a POST request to this endpoint with the image URL in the request body as a JSON object. Example: {"url": "https://example.com/image.jpg"}

3. GET /pred/<string:name>: Send a GET request to this endpoint with the name of the food item as a path parameter to retrieve its details from the database. Example: http://localhost:5000/pred/kale

4. PUT /pred/<string:name>: Send a PUT request to this endpoint with the name of the food item as a path parameter and the updated details as a JSON object in the request body to update the details of the food item in the database. Example: http://localhost:5000/pred/kale with request body {"Category": "kale", "Calories": "50", "Fat": "1", "Protein": "3"}

5. DELETE /pred/<string:name>: Send a DELETE request to this endpoint with the name of the food item as a path parameter to delete the food item from the database. Example: http://localhost:5000/pred/kale
