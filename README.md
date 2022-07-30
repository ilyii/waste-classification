Website for our waste classification!

In the folder jupyter_notebooks/ you'll find the code we used for creating our models...

To run docker:

1. Open path of project in cmd

2. docker build --tag <container-name> .
!Important: Add . at the end of the line!

3. docker images
-> View your images

4. docker run -d -p 5000:5000 <container-name>
-> After successfully build you can open localhost/5000 to use the website

Extra: docker ps
-> Use for see all running docker images atm

Extra: docker stop <container-name>

Extra: docker container prune
-> Stop all unused resources, freeing up space



!!!IMPORTANT!!!
If you want to use the feature to upload images to Google Drive via website:

1. Create mycreds.txt in the same path where app.py is located

2. In mycreds.txt write:
{"access_token": "<YOUR_ACCESS_TOKEN>", "client_id": "<YOUR_CLIENT_ID>", "client_secret": "<YOUR_CLIENT_SECRET>", "refresh_token": "<YOUR_REFRESH_TOKEN>", "token_expiry": "<YOUR...>", "token_uri": "https://oauth2.googleapis.com/token", "user_agent": null, "revoke_uri": "https://oauth2.googleapis.com/revoke", "id_token": null, "id_token_jwt": null, "token_response": {"access_token": "<YOUR...>", "expires_in": 3599, "scope": "https://www.googleapis.com/auth/drive", "token_type": "Bearer"}, "scopes": ["https://www.googleapis.com/auth/drive"], "token_info_uri": "https://oauth2.googleapis.com/tokeninfo", "invalid": false, "_class": "OAuth2Credentials", "_module": "oauth2client.client"}

---> You have to create this via the Google Cloud API!


If you do not want to use this feature, then you do not need to create the file. However, you must then make a change in the code app.py! There you have to comment out the lines: ... must be commented out.
Now no more images are uploaded, so the first function doesn't work anymore. The function to classify a picture now works correctly, but the photo is not uploaded anymore!

