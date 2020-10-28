from flask import Flask
from app import views


app=Flask(__name__)

app.add_url_rule('/','home',views.home,)
app.add_url_rule('/','/',views.home,)
app.add_url_rule('/faceapp','faceapp',views.faceapp)
app.add_url_rule('/faceapp/gender','gender',views.gender,methods=['GET','POST'])
app.add_url_rule('/faceapplive','faceapplive',views.faceapplive)

if __name__ == "__main__":
    app.run(debug=True)
