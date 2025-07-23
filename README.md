# RAG AI For FAA Documents

I am training to be a private pilot and this endevour requires reading a lot of documents

* Books on the specifics of a plane (POHs or Pilot Operating Handbooks)
* Documents about standards required for passing the certificate (FARs or Federal Aviation Regulations, ACS or Airman Certification Standards)
* Textbooks on Flying Techinques or Knowledge (PHAK or Pilot Handbook of Aeronautical Knowledge and AFH or Airplane Flying Handbook)

It would be nice at times to be able to just query these things. Especially things from documents like an FAR since it is concise and gives clear Yes/Nos on what to do or what is required in certain situations. I thought it would be nice to do this with an LLM that would have context of these documents and so I'm trying to build a RAG (Retrieval-Augmented Generation) Agent for this purpose. Mainly just to learn about this flavor of AI and see how it works.

# Learnings so Far

* It's not that hard to run a model locally assuming you have the hardware. I can run Q4 verisons of models locally with a 4080 pretty well.
* It's pretty easy to spend money on OpenAI's API during testing when you have a lot of documents
  * Token counts can get pretty high quickly when you have long documents like we do here
* Evaluating a Model feels a bit underdeveloped
  * Compared with testing frameworks that non-AI programmers have the AI world feels like the Wild West. Working with something non-deterministic by nature and trying to figure out if it answered something well is an interesting problem. I haven't built any evaluation things yet but have read some methods so it seems possible but somewhat difficult to execute well without any subjectivity
* Building a model is like building a full data pipeline
  * There's so much more to RAG than just choosing a model and uploading documents you can
    * Choose how to split the documents into chunks
    * Finetune parameters for how to represent those chunks as vectors
    * Do query construction to take a users query and turn it into multiple
    * Provide metadata for each document to have the underlying LLM model only choose specific documents
      * You can play around with how you define that metadata and the prompt for choosing the appropriate document
    * Change the underlying LLM Model
    * Change the model for creating vectors from chunks
  * It's a bit overwhelming how many things can be changed and if you don't have an evaluation method it's easy to get lost on how these changes matter

# Things to Learn More About/Improve

* Objective ways to evaluate models quickly
* How different changes in the full model can affect accuracy
* More accurate document choosing and accuracy

# TODO

* Figure out the cause of extremely low similarity scores
  * Currently docs are sorted correctly but similarity scores are completely off and near 0
  * Querying Chroma DB returns sane similarity scores (50-70) so likely something not right with the llama_index integration
* Change ChromaDB to client server model
  * Avoids restarting the DB
* Add more documents
* Add evaluation methods