Overview

This project integrates advanced AI functionalities using Streamlit and various LangChain modules, leveraging GroqCloud's API to deliver intelligent and seamless interactions.

Steps to Set Up and Run the Project

Step 1: Install Required Libraries
Before running the code, install the necessary dependencies by executing the following command in your terminal:

pip install streamlit langchain-community langchain langchain-core faiss-cpu langchain-groq PyPDF2 Pillow

Step 2: Obtain a GroqCloud API Key
To access one API key from GroqCloud as part of the free plan, follow these steps:

Go to the GroqCloud homepage.
In the sidebar, select the "Developers" dropdown and click on "Free API Key".
Create an account if you don’t already have one.
Once logged in, navigate to the API keys section to manage your API keys.
Generate a new API key:
Provide a name for your API key.
Generate and copy it to a safe location.

Step 3: Add the Logo
Download the  logo from your local computer and place it in the project's directory. Ensure the file path to the logo is correctly referenced in the code.

Running the Project
Once the above steps are completed:

Launch the Streamlit application with:

streamlit run code.py