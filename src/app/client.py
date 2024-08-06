import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('IMAGE','rb')})

print(resp.json())