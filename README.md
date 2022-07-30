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
