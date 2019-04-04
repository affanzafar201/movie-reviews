# Predict Sentiment Analysis

## Run API
The following command runs the flask api on 5000 port 

```
 python3 app.py
```

### Request for the POST method of api

```
 curl -X POST -H "Content-Type:application/json" -d '{"movie_name":"Inside Out",
"movie_review":"I've noticed a plethora of negative reviews all saying the same thing- their kids did not enjoy/ understand the movie and it was too dark. Yet they also say that the movie was well made and had an excellent concept. My question to these people is this- If you wanted to see a happy meaningless movie with adult toilet humour, why didn't you take your kids to see Minions? Inside out is for those who like quality cinema, to be entertained as well as taught challenging concepts. It is an artistic film. Why do you people always have to assume that all animated films have to be for kids and filled with stupid toilet humour? This is Pixar we are talking about, those who have created Wall-E and Up. Review this movie on its own merits, rather than be biased in your judgement that the film was too dark and confusing for kids. It was never marketed as a film for toddlers and little kids. I went with my brother who is in high school and it was one of the best experiences we had in the cinema in a very long time. Kudos to Pixar; it was truly an amazing, advanced conceptual, artistic film."
}' http://127.0.0.1:5000/predict
```

### Response from api

```
{"movie_review":"I've noticed a plethora of negative reviews all saying the same thing- their kids did not enjoy/ understand the movie and it was too dark. Yet they also say that the movie was well made and had an excellent concept. My question to these people is this- If you wanted to see a happy meaningless movie with adult toilet humour, why didn't you take your kids to see Minions? Inside out is for those who like quality cinema, to be entertained as well as taught challenging concepts. It is an artistic film. Why do you people always have to assume that all animated films have to be for kids and filled with stupid toilet humour? This is Pixar we are talking about, those who have created Wall-E and Up. Review this movie on its own merits, rather than be biased in your judgement that the film was too dark and confusing for kids. It was never marketed as a film for toddlers and little kids. I went with my brother who is in high school and it was one of the best experiences we had in the cinema in a very long time. Kudos to Pixar; it was truly an amazing, advanced conceptual, artistic film.","prob":"0.76212955","sentiment":"Good"}

```
