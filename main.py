import streamlit as st
from pymongo import MongoClient
import urllib, io, json
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from os import getenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = getenv("GROQ_API_KEY")
MONGO_URI = getenv("MONGO_URI")

# Load LLM
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.0,
    api_key=GROQ_API_KEY
)

# MongoDB client
client = MongoClient(MONGO_URI)
db = client["sample_analytics"]

# UI
st.title("Talk to MongoDB (`sample_analytics`)")
st.markdown("Ask questions about `accounts`, `customers`, or `transactions`")

user_input = st.text_area("Your question:")

# --- Schema-aware prompt ---
query_prompt = PromptTemplate(
    input_variables=["question", "sample"],
    template="""
You are a MongoDB AI assistant. Your task is to output ONLY a valid JSON object with:
{{
  "collection": "<collection_name>",
  "query": [ aggregation_pipeline_array ]
}}

Valid collections in the `sample_analytics` database:

1. `accounts`
   - `account_id` (int)
   - `limit` (number): amount of money in the account
   - `products` (array of strings): may include "InvestmentStock", "InvestmentFund", "Derivatives", "Commodity"
   - `customer_id` (string)

2. `customers`
   - `username` (string): register number
   - `name` (string)
   - `email` (string)
   - `birthdate` (date)
   - `address` (object): {{ "street": ..., "city": ..., "state": ..., "zip": ..., "country": ... }}
   - `accounts` (array of account_ids)

3. `transactions`
   - `_id`
   - `account_id` (int)
   - `transaction_count` (int)
   - `bucket_start_date` (ISODate)
   - `bucket_end_date` (ISODate)
   - `transactions`: array of nested transactions (up to 66)

RULES:
- Only return a valid JSON object.
- Never return explanations or extra text.
- Use $match for filters like amount > 1000.
- Use numeric filters, not "$1000" strings.

Example:
Q: List all transactions above $1000
A:
{{
  "collection": "transactions",
  "query": [
    {{ "$match": {{ "transactions.amount": {{ "$gt": 1000 }} }} }}
  ]
}}

Now answer:
Question: {question}
Sample: {sample}
"""
)


# Prompt to convert results into natural language
final_prompt = PromptTemplate(
    template="""
You have a MongoDB question and its query output below. Convert the results into a simple natural language answer.
Only include the answer, nothing else.

Question: {question}
Query Output: {results}
Answer:
""",
    input_variables=["question", "results"]
)

llmchain = LLMChain(llm=llm, prompt=query_prompt, verbose=True)

if user_input:
    if st.button("Submit"):
        with st.spinner("Generating MongoDB query..."):
            response = llmchain.invoke({
                "question": user_input,
                "sample": "Q: Show top 5 accounts by limit.\nA: {\"collection\": \"accounts\", \"query\": [{ \"$sort\": { \"limit\": -1 } }, { \"$limit\": 5 }]}"
            })

        try:
            # Show raw LLM output for debugging
            st.subheader("LLM Response")
            st.code(response["text"], language="json")

            # Parse output
            result = json.loads(response["text"])
            collection_name = result["collection"]
            query = result["query"]

            # Check collection exists
            if collection_name not in db.list_collection_names():
                st.error(f"Collection not found: `{collection_name}`")
            else:
                collection = db[collection_name]
                results = list(collection.aggregate(query))

                if not results:
                    st.warning("No results found for this query.")
                    sample_docs = list(collection.find().limit(3))
                    st.markdown("Sample documents from this collection:")
                    st.json(sample_docs)
                else:
                    for i, result_doc in enumerate(results, start=1):
                        final_chain = LLMChain(llm=llm, prompt=final_prompt)
                        response = final_chain.invoke({
                            "question": user_input,
                            "results": result_doc
                        })
                        st.markdown(f"**🔹 Result {i}:** {response['text']}")

        except Exception as e:
            st.error("Failed to parse or run the query.")
            st.code(response["text"], language="json")
            st.exception(e)
