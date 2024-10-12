from flask import Flask
from app import views

app = Flask(__name__) # webserver gateway interphase (WSGI)

app.add_url_rule(rule='/',endpoint='home',view_func=views.index)
app.add_url_rule(rule='/app/',endpoint='app',view_func=views.app)
app.add_url_rule(rule='/calculator/',endpoint='calculator',view_func=views.calculator)
app.add_url_rule(rule='/DOIschain/',endpoint='DOIschain',view_func=views.DOIschain)

#app.add_url_rule(rule='/app/trading/',endpoint='trading',view_func=views.tradingapp2)
app.add_url_rule(rule='/app/trading/',
                 endpoint='trading',
                 view_func=views.tradingapp2,
                 methods=['GET','POST'])
app.add_url_rule(rule='/app/gender/',
                 endpoint='gender',
                 view_func=views.genderapp,
                 methods=['GET','POST'])
app.add_url_rule(rule='/rif/',endpoint='rif',view_func=views.rifapp,
                 methods=['GET','POST'])
app.add_url_rule(rule='/certificate/',endpoint='certificate',view_func=views.app_certificates,
                 methods=['GET','POST'])

if __name__ == "__main__":
    app.run(debug=True)

#adicion