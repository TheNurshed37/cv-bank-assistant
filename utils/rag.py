# rag.py 
import os
import getpass
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import re

from utils.config import settings
from utils.logger import logger

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

def rag_answer(question: str) -> str:
    """Answer questions using RAG with RELEVANT file path source attribution"""
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
        
        # Case 1: No documents retrieved
        if not retrieved_docs:
            return "I couldn't find any relevant information in the uploaded CVs to answer your question. Please try asking about specific skills, experiences, or candidates that might be in the database."

        # Prepare context and track which candidates are actually used
        context_parts = []
        candidate_usage_tracker = {}  # Track which candidates provide relevant content
        
        # Import candidate manager to get source paths
        from utils.candidate_manager import CandidateManager
        candidate_manager = CandidateManager("hash_index.json")
        
        for doc in retrieved_docs:
            candidate_name = doc.metadata.get('candidate_name', 'Unknown')
            candidate_id = doc.metadata.get('candidate_id')
            
            context_parts.append(f"CANDIDATE: {candidate_name}\nCONTENT: {doc.page_content}")
            
            # Store candidate info for later source path resolution
            if candidate_id:
                candidate_usage_tracker[candidate_id] = {
                    'candidate_name': candidate_name,
                    'doc_content': doc.page_content
                }
        
        context_text = "\n\n".join(context_parts)
        
        # Prepare prompt - REFINED FOR HR ASSISTANT
        prompt_template = """
        You are an expert HR Assistant specialized in CV analysis and candidate screening. Your role is to help HR professionals quickly identify the best candidates based on their resumes.

        CONTEXT FROM UPLOADED CVs:
        {context}

        QUESTION: {question}

        CRITICAL INSTRUCTIONS FOR CV ANALYSIS:
        1. **STRICT CONTEXT-BASED ANSWERS**: Only use information from the provided CV context. Do not hallucinate or assume missing details.

        INSTRUCTIONS:
        1. Base your answer **only** on the given CV context — never assume or fabricate details.
        2. Always mention the **candidate's name** when referring to their details.
        3. If the context does not contain the requested information, clearly state:
        → "The provided CVs do not include that information."
        4. Be specific when describing:
        - Skills and technical proficiencies
        - Work experience and job roles
        - Education and qualifications
        - Projects, achievements, or certifications
        5. If multiple candidates are present, compare or list them clearly.
        6. If information is incomplete for a candidate, specify what **is available**.
        7. When appropriate, reference the candidate’s file path 
        8. Keep the answer **concise, factual, and well-organized**.
        9. Complete your response without truncation or repetition.
        10. Keep the answer short and to the point.

        ANSWER:
        """
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            max_tokens=2000,  # Increased for more comprehensive analysis
            convert_system_message_to_human=True
        )
        
        final_prompt = prompt_template.format(context=context_text, question=question)
        response = llm.invoke(final_prompt)
        
        answer = response.content.strip()
        
        # Clean up formatting
        answer = re.sub(r'[*_`#]', '', answer)
        answer = re.sub(r'\n+', '\n', answer).strip()
        
        # Case 2: Check for insufficient information
        if is_insufficient_answer(answer, retrieved_docs, question):
            return "Based on the available CVs, I don't have enough information to answer this specific question. The uploaded resumes don't contain the details you're looking for."
        
        # Case 3: Check for empty response
        if not answer or len(answer) < 20:
            return "I couldn't generate a response based on the available CV information. Please try rephrasing your question or ask about different aspects of the candidates."
        
        # Add ONLY RELEVANT source paths based on the actual answer
        if candidate_usage_tracker:
            relevant_source_paths = set()
            
            # Analyze which candidates were actually mentioned in the answer
            answer_lower = answer.lower()
            
            for candidate_id, candidate_info in candidate_usage_tracker.items():
                candidate_name = candidate_info['candidate_name']
                candidate_name_lower = candidate_name.lower()
                
                # Check if this candidate is mentioned in the answer
                if candidate_name_lower in answer_lower:
                    # Get the source path for this relevant candidate
                    candidate_data = candidate_manager.get_candidate_by_id(candidate_id)
                    if candidate_data and candidate_data.get('source_path'):
                        source_path = candidate_data['source_path']
                        relevant_source_paths.add(source_path)
                        logger.info(f"Added relevant source path for {candidate_name}: {source_path}")
            
            # If no specific candidates mentioned, include all from context as fallback
            if not relevant_source_paths:
                logger.info("No specific candidates mentioned in answer, using all retrieved sources")
                for candidate_id in candidate_usage_tracker.keys():
                    candidate_data = candidate_manager.get_candidate_by_id(candidate_id)
                    if candidate_data and candidate_data.get('source_path'):
                        source_path = candidate_data['source_path']
                        relevant_source_paths.add(source_path)
            
            # Add relevant source attribution
            if relevant_source_paths:
                sorted_paths = sorted(list(relevant_source_paths))
                paths_str = ', '.join(sorted_paths)
                answer += f'\n\n[source: {paths_str}]'
                logger.info(f"Added RELEVANT source paths: {paths_str}")
        
        return answer
        
    except Exception as e:
        logger.error(f"RAG failed: {e}")
        return "Error analyzing CVs. Please try again."


def is_insufficient_answer(answer: str, retrieved_docs: list, question: str) -> bool:
    """Check if the answer indicates insufficient information"""
    insufficient_phrases = [
        "i don't know", "i cannot answer", "no information", 
        "not mentioned", "not provided", "not available",
        "based on the context", "the context doesn't"
    ]
    
    answer_lower = answer.lower()
    
    # Check for insufficient information phrases
    if any(phrase in answer_lower for phrase in insufficient_phrases):
        return True
    
    # Check if answer is too generic compared to question
    if len(answer.split()) < 10 and len(question.split()) > 5:
        return True
        
    return False

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
    """
    Simplified LLM-driven approach:
    - No documents: Use general knowledge LLM
    - With documents: ALWAYS use RAG (no intent detection)
    """
    try:
        # Check if FAISS index exists and has documents
        faiss_exists = os.path.exists("faiss_index") and os.listdir("faiss_index")
        
        if not faiss_exists:
            # Scene 1: No documents uploaded - use general knowledge
            logger.info("No documents - using general knowledge LLM")
            return llm_general_knowledge(question)
        
        # Scene 2: Documents uploaded - ALWAYS use RAG
        logger.info("Documents available - using RAG only")
        return rag_answer(question)

    except Exception as e:
        logger.error(f"Error in answer_question: {e}", exc_info=True)
        return "I encountered an error while processing your question. Please try again."

# For backward compatibility
def answer_question_original(question: str):
    return answer_question(question)