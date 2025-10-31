# rag.py 
import os
import getpass
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import re

from config import settings
from logger import logger

load_dotenv()

def llm_general_knowledge(question: str) -> str:
    """
    Answer general knowledge questions using LLM
    """
    try:
        prompt_template = """
        You are a helpful AI assistant. Answer the following question clearly and accurately.
        
        QUESTION: {question}
        
        GUIDELINES:
        - Provide a clear, factual answer
        - If you cannot answer, explain why briefly
        - Keep it informative but concise
        - Use plain text without markdown
        - Be helpful and professional
        
        ANSWER:
        """
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            max_tokens=500,
            convert_system_message_to_human=True
        )
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question"]
        )
        
        final_prompt = prompt.invoke({"question": question})
        response = llm.invoke(final_prompt)
        
        answer = response.content.strip()
        
        # Clean up formatting
        answer = re.sub(r'[*_`#]', '', answer)
        answer = re.sub(r'\n+', '\n', answer)
        answer = answer.strip()
        
        return answer
        
    except Exception as e:
        logger.error(f"General knowledge LLM failed: {e}")
        return "I encountered an error while processing your question. Please try again."

def llm_determine_intent(question: str) -> str:
    """
    Let LLM decide if the question is about uploaded CVs or general knowledge
    Returns: "cv_related" or "general_knowledge"
    """
    try:
        prompt_template = """
        You are analyzing a user question to determine if it's about analyzing CVs/resumes that have been uploaded to the system.
        
        QUESTION: {question}
        
        DECISION CRITERIA:
        - If the question is about candidates, resumes, CVs, skills, experience, education, work history, hiring, or comparing applicants → "cv_related"
        - If the question is general knowledge, technical concepts, or anything not related to analyzing uploaded documents → "general_knowledge"
        
        Return ONLY one of these two words: "cv_related" or "general_knowledge"
        
        DECISION:
        """
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.1,
            max_tokens=50,
            convert_system_message_to_human=True
        )
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question"]
        )
        
        final_prompt = prompt.invoke({"question": question})
        response = llm.invoke(final_prompt)
        
        intent = response.content.strip().lower()
        
        if "cv_related" in intent:
            logger.info(f"LLM determined intent: CV-related for question: {question}")
            return "cv_related"
        else:
            logger.info(f"LLM determined intent: General knowledge for question: {question}")
            return "general_knowledge"
            
    except Exception as e:
        logger.warning(f"LLM intent detection failed, defaulting to general knowledge: {e}")
        return "general_knowledge"

def rag_answer(question: str) -> str:
    """Answer questions using RAG from uploaded CVs - SIMPLIFIED"""
    try:
        # Initialize
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            GOOGLE_API_KEY = getpass.getpass("Enter Google API key: ")
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

        embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
        
        # Retrieve documents
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(question)
        
        if not retrieved_docs:
            return "No relevant information found in the uploaded CVs."

        # Prepare context
        context_parts = []
        candidate_names = []
        
        for doc in retrieved_docs:
            candidate_name = doc.metadata.get('candidate_name', 'Unknown')
            candidate_names.append(candidate_name)
            context_parts.append(f"CANDIDATE: {candidate_name}\nCONTENT: {doc.page_content}")
        
        context_text = "\n\n".join(context_parts)
        

        prompt_template = """
        You are an HR assistant analyzing CVs. Answer the question based STRICTLY on the provided context clearly and accurately.
        
        CONTEXT FROM UPLOADED CVs:
        {context}
        
        QUESTION: {question}
        
        CRITICAL INSTRUCTIONS:
        1. You MUST provide a clear, factual answer answer based on the context above, Keep it informative but concise
        2. If you cannot answer, explain why briefly, List names if asked for names
        3. Be specific about skills, experience, education, and projects
        4. If asking about specific roles (like SQA, Frontend, Java developer, UI/UX, Python developer etc), analyze their relevant experience and projects
        5. Reference candidate names specifically
        6. If information is limited for a candidate, state what information IS available
        7. If comparing, be brief and clear
        8. Be helpful and professional
        9. Complete your answer without truncation

        FOR CANDIDATE LISTING QUESTIONS:
        - List each candidate by name
        - Provide a clear summery of their background and key qualifications
        - Highlight relevant skills and experience for the asked role
        - Be specific and detailed
        
        ANSWER:
        """
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            max_tokens=1500,  # Reduced to avoid truncation
            convert_system_message_to_human=True
        )
        
        final_prompt = prompt_template.format(context=context_text, question=question)
        response = llm.invoke(final_prompt)
        
        answer = response.content.strip()
        
        # Clean up
        answer = re.sub(r'[*_`#]', '', answer)
        answer = re.sub(r'\n+', '\n', answer).strip()
        
        # Better fallback check
        if not answer or len(answer) < 20 or answer.endswith(('1.', '2.', '-', '•')):
            return generate_smart_fallback(candidate_names, retrieved_docs, question)
        
        return answer
        
    except Exception as e:
        logger.error(f"RAG failed: {e}")
        return "Error analyzing CVs. Please try again."

def generate_smart_fallback(candidate_names, retrieved_docs, question):
    """Generate smart fallback responses"""
    try:
        # Simple name list for "just name" questions
        if 'just name' in question.lower() or 'only name' in question.lower():
            return f"Candidates: {', '.join(candidate_names)}"
        
        # Role-specific responses
        if any(role in question.lower() for role in ['sqa', 'qa', 'testing', 'quality']):
            return f"SQA candidates: {', '.join(candidate_names)}"
        elif any(role in question.lower() for role in ['frontend', 'react', 'javascript']):
            return f"Frontend candidates: {', '.join(candidate_names)}"
        elif any(role in question.lower() for role in ['backend', 'python', 'java']):
            return f"Backend candidates: {', '.join(candidate_names)}"
        
        # Default fallback
        return f"Found these candidates: {', '.join(candidate_names)}. Ask about specific skills or roles for details."
        
    except Exception:
        return f"Candidates found: {', '.join(candidate_names)}"


def answer_question(question: str) -> str:
    """al know
    Simplified LLM-driven approach:
    - No documents: Always use general knowledge
    - With documents: Let LLM decide if question is CV-related or general knowledge
    """
    try:
        # Check if FAISS index exists and has documents
        faiss_exists = os.path.exists("faiss_index") and os.listdir("faiss_index")
        
        if not faiss_exists:
            # Scene 1: No documents uploaded - always use general knowledge
            logger.info("No documents - using general knowledge LLM")
            return llm_general_knowledge(question)
        
        # Scene 2: Documents uploaded - let LLM decide routing
        logger.info("Documents available - LLM deciding routing")
        intent = llm_determine_intent(question)
        
        if intent == "cv_related":
            # Scene 2.1: CV-related question - use RAG
            logger.info("Using RAG for CV-related question")
            return rag_answer(question)
        else:
            # Scene 2.2: General knowledge question - use LLM directly
            logger.info("Using general knowledge LLM")
            return llm_general_knowledge(question)

    except Exception as e:
        logger.error(f"Error in answer_question: {e}", exc_info=True)
        return "I encountered an error while processing your question. Please try again."

# For backward compatibility
def answer_question_original(question: str):
    return answer_question(question)