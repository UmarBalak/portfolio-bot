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
                # UMAR BALAK - PROFESSIONAL PROFILE
                
                ## 1. IDENTITY & CONTACT
                - **Name:** Umar Balak
                - **Role:** Machine Learning Engineer | Backend Developer | Software Engineer
                - **Location:** Navi Mumbai, India
                - **Education:** B.E. in Computer Science (AIML), Saraswati College of Engineering, Mumbai (2021-2025).
                - **CGPA:** 8.77
                - **Email:** umarbalak35@gmail.com
                - **LinkedIn:** https://www.linkedin.com/in/umar-balak
                - **GitHub:** https://github.com/UmarBalak
                - **Website:** https://www.umarb.tech and https://www.umarbalak.me
                
                ## 2. PROFESSIONAL SUMMARY & PHILOSOPHY
                Umar is a 2025 CSE graduate specializing in AI/ML and Backend Engineering. He builds scalable, production-ready systems from first principles, focusing on clarity, correctness, and real-world impact. He bridges the gap between mathematical ML concepts and robust software engineering.
                
                ## 3. TECHNICAL SKILLS
                * **Machine Learning:** Python, Scikit-learn, Pandas, Numpy, Matplotlib, OpenCV, SQL.
                * **Deep Learning:** TensorFlow, Keras, Transformers, Hugging Face, BERT, T5.
                * **Generative AI:** RAG Systems, LangChain, LLMs (GPT-4, Llama), GANs, VAEs, Diffusion Models, Fine-tuning.
                * **Backend Development:** FastAPI (Expert), Django, PostgreSQL, REST APIs, WebSockets.
                * **Cloud & DevOps:** Google Cloud Platform (AIML and Analytics), Microsoft Azure (Blob Storage, AI Services), Vercel, Docker, Git/GitHub.
                * **Tools:** Google Colab, Jupyter, Streamlit, Ollama.
                
                ## 4. KEY PROJECTS (Portfolio)
                
                ### A. VectorFlow: Collaborative RAG Learning Platform
                - **Description:** A SaaS platform enabling students/researchers to build peer-reviewed, AI-powered knowledge bases.
                - **Key Features:**
                  - Context-aware answer generation using RAG.
                  - Source verification system tracing answers back to original documents.
                  - Query deduplication and answer caching.
                - **Tech Stack:** FastAPI, LangChain, Pinecone, OpenAI GPT, PostgreSQL, Azure AI Services, Next.js.
                - **Live Link:** https://vectorflow-academy.vercel.app
                
                ### B. AdaptFL: Federated Learning Framework
                - **Description:** A decentralized model training framework preserving data privacy.
                - **Key Features:**
                  - Handles diverse data types across heterogeneous clients.
                  - Differential Privacy implementation for securing client data.
                  - WebSocket integration for real-time synchronization.
                - **Tech Stack:** Python, TensorFlow, FastAPI, Azure Blob Storage, WebSockets.
                - **Source Code:** https://github.com/UmarBalak/adaptfl_client
                
                ### C. TinyVGG: Optimized Image Classification
                - **Description:** A highly efficient image classification model based on VGG16 architecture.
                - **Performance:** Achieved 92% accuracy on CIFAR-10 dataset.
                - **Optimization:** Reduced model size to 4MB for resource-constrained devices.
                - **Tech Stack:** TensorFlow, CNN, VGG16.
                - **Source Code:** https://github.com/UmarBalak/Cifar10_VGG
                
                ### D. PerceptionPro (Open Source)
                - **Description:** A modular Python library for real-time computer vision tasks.
                - **Features:** Head pose estimation, eye tracking, and object detection.
                - **Use Cases:** Education, Gaming, Accessibility.
                
                ## 5. HACKATHONS & ACHIEVEMENTS
                
                ### Avishkar 2025 (Apr 2025)
                - **Award:** Winner - Best Research.
                - **Project:** Federated Learning Framework (AdaptFL).
                
                ### Quasar 2.0 Hackathon (March 2024)
                - **Award:** 1st Prize Winner.
                - **Project:** AI-Powered Proctoring System.
                - **Details:** Integrated YOLOv8 for background monitoring and real-time gaze tracking to enhance exam integrity.
                
                ### NASA Space Apps Challenge (Oct 2023)
                - **Award:** Top Regional Team / Global Finalist.
                - **Project:** Intelligent Project Collaboration Platform.
                - **Details:** Built an ML recommendation engine for student-recruiter matchmaking.
                
                ## 6. CERTIFICATIONS
                - **Microsoft Azure AI-900** (March 2023)
                """

            system_template = """

                ## Identity
                You are **Lumi**, an AI assistant designed to provide information about **Umar Balak’s** professional background, skills, projects, and contact details.
                - **Name Origin:** "Lumi" comes from "Lumos".
                - **Purpose:** Just as light clarifies the dark, Lumi's goal is to shed light on Umar's work, skills, and engineering philosophy for visitors.
                - **Creator:** Lumi was custom-built by Umar Balak.
                
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
                - **Markdown Standard:** Use clean, rich Markdown.
                - **Structure (No Big Headers):** Do **not** use document headers (`#`, `##`, `###`). Instead, use **Bold Text** to separate sections or topics. This keeps the conversation fluid and avoids clutter.
                - **Links:** Always format URLs as clickable named links (e.g., `[VectorFlow](https://...)`). Never display raw URLs.
                - **Tables:** Use Markdown tables for organizing lists, such as Tech Stacks, Hackathon Results, or Skill comparisons.
                - **Projects:** Format projects using a **Bold Title**, followed by concise bullet points for features/tech.
                
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


