from wtforms import Form, validators, StringField, IntegerField


class InputFormWord2Vec(Form):
    words = StringField(label='Odia words', default="ଗଛ, ସଙ୍ଗୀତ, ଚଳଚ୍ଚିତ୍ର", validators=[validators.InputRequired()])
    num_neighbours = IntegerField(label='Number of neighbours', default=5, validators=[validators.InputRequired()])
