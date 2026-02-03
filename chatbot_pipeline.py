from typing import Dict, Any
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv
import logging

from llm_models import LLM

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotPipeline:
    """
    """

    def __init__(self, llm_model: str):

        self.llm_model = llm_model

        # Initialize components
        self.llm = LLM(llm_model=self.llm_model, max_messages=10)

    def query_with_template_method(self,
                                  query_text: str,
                                  temperature: float = 0.5) -> Dict[str, Any]:
        """
        """
        try:

            user_info = """
            Name: Umar Balak
            Role: Software Engineer (Backend & AI/ML/GenAI)
            GitHub: https://github.com/UmarBalak
            Linkedin: https://www.linkedin.com/in/umar-balak/
            Email: umarbalak35@gmail.com
            """

            system_template = """
                ## Identity
                You are **Lumi**, an AI assistant designed to provide information about **Umar Balak’s** professional background, skills, projects, and contact details.
                
                ## Knowledge Source
                - Use the provided **`user_info`** as the single source of truth.
                - Answer questions **only** when they relate directly to Umar’s professional profile, experience, or work.
                - Do not infer, assume, or invent any information that is not explicitly present in `user_info`.
                
                ## Scope
                - This assistant is focused exclusively on Umar’s portfolio and career-related information.
                - If a question is unrelated to Umar’s professional background, respond that you are designed to answer questions about his portfolio only.

                ## Missing Information Handling
                - If a requested detail is not available in `user_info`, state that the information is not currently available.
                - Suggest contacting Umar directly via the email or LinkedIn details provided in `user_info`.

                ## Hiring and Collaboration
                - If a user expresses interest in hiring, collaboration, or professional opportunities, share the relevant contact information from `user_info`.
                - Maintain a professional and encouraging tone.

                ## Style and Tone
                - Professional, concise, and helpful.
                - Refer to Umar in the third person as “Umar” or “he”.
                - Avoid unnecessary preamble. Go directly to the answer.

                ## Response Formatting
                - Use clean, readable **Markdown**.
                - Simple questions: respond in 1–2 sentences.
                - Complex questions: use clear structure with bullet points where appropriate.
                
                """
                            

            human_template = """
            ## Context Data:
            {user_info}

            ## User Query:
            {query_text}

            """


            system_prompt = SystemMessagePromptTemplate.from_template(system_template)
            human_prompt = HumanMessagePromptTemplate.from_template(human_template)
            chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

            logging.info("Prompt created. Invoking LLM...")

            # Use the new template method that preserves memory
            response = self.llm.invoke_with_template(
                chat_prompt,
                {"query_text": query_text, "user_info": user_info}
            )

            logging.info(response)

            # Extract the answer and tokens from the AIMessage object
            answer = response.get("content", "")
            tokens_used = response.get("usage_metadata", {})

            # Return enhanced response with backward compatibility
            return {
                "answer": answer,
                "tokens_used": tokens_used,
                "query_text": query_text,
            }

        except Exception as e:
            logger.error(f"Error during enhanced query processing: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    # Initialize LLM with memory (using GPT-5 by default)
    llm_with_memory = LLM(llm_model="kimi")

    chatbot_pipeline = ChatbotPipeline(llm_model="kimi")
    
    # Example conversation
    try:
        result = chatbot_pipeline.query_with_template_method(
            query_text="Can you summarize Umar Balak's GitHub profile?"
        )
        print("Result:", result)

        result = chatbot_pipeline.query_with_template_method(
            query_text="What are his main skills and projects?"
        )
        print("Result:", result)
        result = chatbot_pipeline.query_with_template_method(
            query_text="Can you provide summary of our previous discussion?"
        )
        print("Result:", result)

    except Exception as e:
        print(f"Error: {e}")