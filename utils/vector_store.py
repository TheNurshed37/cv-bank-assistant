import os
import re
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

from utils.config import settings
from utils.logger import logger

def initialize_docling_converter():
    """Initialize Docling converter for optimal CV parsing - OPTIMIZED"""
    pipeline_options = PdfPipelineOptions(
        do_table_structure=False,  # DISABLED for speed
        do_ocr=False,
        generate_page_images=False,
        generate_picture_images=False,
        do_ocr_strategy="never",  # Explicitly disable OCR
    )
    
    doc_converter = DocumentConverter()
    return doc_converter

def extract_structured_sections_with_docling(docling_result):
    """
    Use Docling's native structure to identify CV sections 
    """
    structured_sections = {}
    
    try:
        # SIMPLIFIED: Just use markdown export and basic section detection
        full_markdown = docling_result.document.export_to_markdown()
        
        # Basic section detection using markdown headers
        lines = full_markdown.split('\n')
        current_section = "Professional Profile"
        content_buffer = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('#') and len(line) > 2:
                # This is a header line
                if content_buffer:
                    structured_sections[current_section] = '\n'.join(content_buffer)
                    content_buffer = []
                current_section = line.lstrip('#').strip()
            elif line:
                content_buffer.append(line)
        
        # Add the last section
        if content_buffer:
            structured_sections[current_section] = '\n'.join(content_buffer)
            
    except Exception as e:
        logger.warning(f"Docling structure extraction failed: {e}")
        # Ultimate fallback
        try:
            full_text = docling_result.document.export_to_markdown()
            structured_sections["Full Document"] = full_text
        except:
            structured_sections["Full Document"] = "Content extraction failed"
    
    return structured_sections

def rule_based_section_filtering(structured_sections):
    """
    Fast rule-based filtering - KEEP ALL TECHNICAL SECTIONS
    """
    # Keep all sections for now - let LLM handle filtering
    return structured_sections

def remove_personal_info(text):
    """Remove personal information patterns from text """
    if not text:
        return text
        
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    
    # URLs (but keep GitHub/LinkedIn for technical assessment)
    text = re.sub(r'http[s]?://(?!github\.com|linkedin\.com|gitlab\.com)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
    
    # Physical addresses
    text = re.sub(r'\d+\s+[\w\s]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b', '[ADDRESS]', text)
    
    return text

def lightweight_llm_validation(filtered_sections, candidate_name):
    """
    Lightweight LLM validation and final formatting - PRESERVE TECHNICAL DETAILS
    """
    try:
        # Prepare sections for LLM
        sections_text = ""
        for section_name, content in filtered_sections.items():
            if content and content.strip():
                # Only remove obvious personal info, keep technical details
                clean_content = remove_personal_info(content)
                sections_text += f"## {section_name}\n{clean_content}\n\n"
        
        if not sections_text.strip():
            logger.warning("No content for LLM validation")
            return f"## Professional Profile: {candidate_name}\n\nNo relevant professional information extracted."
        
        prompt_template = """
        CV CLEANING AND STRUCTURING ASSISTANT

        CANDIDATE: {candidate_name}

        RAW EXTRACTED SECTIONS:
        {sections_text}

        TASKS:
        1. Remove personal contact information (emails, phones, addresses)
        2. PRESERVE ALL technical skills, programming languages, tools, frameworks
        3. PRESERVE ALL work experience, projects, education details
        4. Keep professional summaries and objectives
        5. Remove completely:
           - Personal hobbies and interests (unless relevant to tech)
           - References
           - Excessive personal details
        6. Structure the content with clear headers
        7. Keep the candidate's name at the top
        8. Make it concise but comprehensive - PRESERVE TECHNICAL DETAILS

        Return ONLY the cleaned, professional CV content as a SINGLE cohesive document.
        Focus on preserving skills, experience, and technical capabilities.

        CLEANED PROFESSIONAL CV:
        """
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.1,
            max_tokens=2000,  # Increased for more content
            convert_system_message_to_human=True
        )
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["candidate_name", "sections_text"]
        )
        
        final_prompt = prompt.invoke({
            "candidate_name": candidate_name,
            "sections_text": sections_text
        })
        
        response = llm.invoke(final_prompt)
        logger.info("Lightweight LLM validation completed")
        
        return response.content
        
    except Exception as e:
        logger.warning(f"LLM validation failed, using rule-based content: {e}")
        # Fallback: minimal cleaning but preserve content
        fallback_content = f"## Professional Profile: {candidate_name}\n\n"
        for section_name, content in filtered_sections.items():
            if content and content.strip():
                clean_content = remove_personal_info(content)
                fallback_content += f"## {section_name}\n{clean_content}\n\n"
        return fallback_content

def create_single_golden_chunk(final_content, candidate_data):
    """Create exactly ONE comprehensive chunk per CV - NO SPLITTING"""
    # Log the actual content length and preview
    logger.info(f"Golden chunk content length: {len(final_content)} characters")
    logger.info(f"Golden chunk preview: {final_content[:200]}...")
    
    # Ensure the content is not too long (safety check)
    if len(final_content) > 15000:  # Increased limit
        logger.warning(f"Content too long ({len(final_content)} chars), truncating")
        final_content = final_content[:15000] + "\n\n[Content truncated for length]"
    
    doc = Document(
        page_content=final_content,
        metadata={
            "source": candidate_data["original_filename"],
            "candidate_id": candidate_data["candidate_id"],
            "candidate_name": candidate_data["candidate_name"],
            "document_type": "cv_golden_chunk",
            "processing_method": "single_chunk_enhanced",
            "chunk_strategy": "single_comprehensive",
            "content_length": len(final_content)  # Track actual length
        }
    )
    
    logger.info(f"Created single golden chunk with {len(final_content)} characters")
    return [doc]

def convert_pdf_with_docling_enhanced(pdf_path, candidate_data):
    """
    Enhanced PDF processing - GUARANTEES SINGLE CHUNK
    """
    logger.info(f"Starting enhanced processing for: {candidate_data['candidate_name']}")
    
    try:
        # Step 1: Docling structural parsing
        converter = initialize_docling_converter()
        result = converter.convert(pdf_path)
        
        # Step 2: Extract structured sections using Docling
        structured_sections = extract_structured_sections_with_docling(result)
        logger.info(f"Docling extracted {len(structured_sections)} sections: {list(structured_sections.keys())}")
        
        # Step 3: Lightweight LLM validation and final cleanup
        final_content = lightweight_llm_validation(structured_sections, candidate_data["candidate_name"])
        
        # Step 4: Create SINGLE golden chunk - NO SPLITTING
        golden_chunks = create_single_golden_chunk(final_content, candidate_data)
        
        logger.info(f"Successfully created {candidate_data['candidate_name']}")
        return golden_chunks
        
    except Exception as e:
        logger.error(f"Enhanced processing failed for {pdf_path}: {e}")
        # Fallback to basic processing with single chunk
        logger.info("Using fallback basic processing with single chunk")
        return convert_pdf_with_docling_basic(pdf_path, candidate_data)

def convert_pdf_with_docling_enhanced_optimized(pdf_path, candidate_data, pre_processed_content):
    """
    Enhanced PDF processing using pre-processed content to avoid double processing
    """
    logger.info(f"Starting optimized processing for: {candidate_data['candidate_name']}")
    
    try:
        # Use the pre-processed content instead of reprocessing the PDF
        full_markdown = pre_processed_content
        
        # Basic section detection using markdown headers
        lines = full_markdown.split('\n')
        current_section = "Professional Profile"
        content_buffer = []
        structured_sections = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('#') and len(line) > 2:
                # This is a header line
                if content_buffer:
                    structured_sections[current_section] = '\n'.join(content_buffer)
                    content_buffer = []
                current_section = line.lstrip('#').strip()
            elif line:
                content_buffer.append(line)
        
        # Add the last section
        if content_buffer:
            structured_sections[current_section] = '\n'.join(content_buffer)
            
        logger.info(f"Extracted {len(structured_sections)} sections from pre-processed content: {list(structured_sections.keys())}")
        
        # Step 3: Lightweight LLM validation and final cleanup
        final_content = lightweight_llm_validation(structured_sections, candidate_data["candidate_name"])
        
        # Step 4: Create SINGLE golden chunk - NO SPLITTING
        golden_chunks = create_single_golden_chunk(final_content, candidate_data)
        
        logger.info(f"Successfully created chunk for {candidate_data['candidate_name']} ")
        return golden_chunks
        
    except Exception as e:
        logger.error(f"Optimized processing failed for {pdf_path}: {e}")
        # Fallback to original processing
        logger.info("Using fallback processing")
        return convert_pdf_with_docling_enhanced(pdf_path, candidate_data)

def convert_pdf_with_docling_basic(pdf_path, candidate_data):
    """Fallback basic processing - ALSO SINGLE CHUNK"""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    raw_text = result.document.export_to_markdown()
    
    # Basic personal info removal
    cleaned_text = remove_personal_info(raw_text)
    
    # Ensure single chunk
    if len(cleaned_text) > 15000:
        cleaned_text = cleaned_text[:15000] + "\n\n[Content truncated]"
    
    doc = Document(
        page_content=cleaned_text,
        metadata={
            "source": pdf_path,
            "candidate_id": candidate_data["candidate_id"],
            "candidate_name": candidate_data["candidate_name"],
            "document_type": "cv_golden_chunk",
            "processed": "basic_single_chunk",
            "content_length": len(cleaned_text)
        }
    )
    
    logger.info(f"Created fallback single chunk with {len(cleaned_text)} characters")
    return [doc]

def add_to_faiss_index(pdf_path, file_hash, faiss_dir, hash_index_file, permanent_storage_path=None):
    """Enhanced with permanent PDF storage and real path storage"""
    
    from utils.candidate_manager import CandidateManager
    candidate_manager = CandidateManager(hash_index_file)
    
    # Check if file already processed
    hash_index = candidate_manager.load_hash_index()
    if file_hash in hash_index["file_hashes"]:
        candidate_id = hash_index["file_hashes"][file_hash]
        candidate_data = hash_index["candidates"][candidate_id]
        return f"'{candidate_data['candidate_name']}' already exists.", 201

    # Use the permanent storage path for source attribution
    source_path_to_store = permanent_storage_path if permanent_storage_path else pdf_path
    logger.info(f"Storing permanent PDF path: {source_path_to_store}")

    # PROCESS PDF
    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        full_content = result.document.export_to_markdown()
        
        sample_content = full_content[:2000]
        
        filename = os.path.basename(pdf_path)
        
        # Register candidate with PERMANENT storage path
        candidate_data = candidate_manager.register_candidate(
            file_hash=file_hash, 
            filename=filename, 
            content=sample_content,
            file_path=source_path_to_store  # Store the permanent, accessible path
        )
        
        docs = convert_pdf_with_docling_enhanced_optimized(pdf_path, candidate_data, full_content)
        
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        filename = os.path.basename(pdf_path)
        candidate_data = candidate_manager.register_candidate(
            file_hash=file_hash, 
            filename=filename, 
            content=None,
            file_path=source_path_to_store
        )
        docs = convert_pdf_with_docling_enhanced(pdf_path, candidate_data)
    
    # Save to vector store
    chunks = docs
    embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    
    if os.path.exists(os.path.join(faiss_dir, "index.faiss")):
        vector_store = FAISS.load_local(faiss_dir, embedding_model, allow_dangerous_deserialization=True)
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embedding_model)

    vector_store.save_local(faiss_dir)
    
    return f"Chunk created successfully for '{candidate_data['candidate_name']}' (ID: {candidate_data['candidate_id']})", 200