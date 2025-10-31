# import uuid
# import re
# import os
# from datetime import datetime
# from typing import Dict, Any, Optional
# import json

# from logger import logger

# class CandidateManager:
#     def __init__(self, hash_index_file: str):
#         self.hash_index_file = hash_index_file
    
#     def generate_candidate_id(self) -> str:
#         """Generate unique UUID for candidate"""
#         return str(uuid.uuid4())
    
#     def extract_name_from_filename(self, filename: str) -> str:
#         """
#         Extract candidate name from filename
#         """
#         try:
#             # Remove file extension
#             name_only = os.path.splitext(filename)[0]
            
#             # Remove common suffixes
#             suffixes = ['_cv', '_resume', '-cv', '-resume', '_CV', '_Resume', ' cv', ' resume']
#             for suffix in suffixes:
#                 name_only = name_only.replace(suffix, '')
            
#             # Remove years and numbers
#             name_only = re.sub(r'[_\-]\d{4}', '', name_only)
#             name_only = re.sub(r'[_\-]\d+', '', name_only)
            
#             # Replace separators with space
#             name_only = re.sub(r'[_\-.]+', ' ', name_only)
            
#             # Title case and clean up
#             name_parts = [part.strip().title() for part in name_only.split() if part.strip()]
#             candidate_name = ' '.join(name_parts)
            
#             # Validate it looks like a name (at least 2 parts)
#             if len(name_parts) >= 2 and all(len(part) > 1 for part in name_parts):
#                 logger.info(f"Name extracted from filename: {candidate_name}")
#                 return candidate_name
#             else:
#                 logger.warning(f"Filename doesn't contain proper name: {filename}")
#                 return filename
                
#         except Exception as e:
#             logger.error(f"Error extracting name from filename {filename}: {e}")
#             return filename
    
#     def extract_name_from_content(self, content: str) -> Optional[str]:
#         """
#         Extract candidate name from CV content using multiple strategies
#         """
#         try:
#             if not content:
#                 return None
            
#             # Strategy 1: Look for name patterns at the beginning of document
#             lines = content.split('\n')
#             for i, line in enumerate(lines[:10]):  # Check first 10 lines
#                 line = line.strip()
#                 if not line:
#                     continue
                
#                 # Common name patterns
#                 # "Md. Mahadi Hasan" or "Mahadi Hasan" or "Hasan, Mahadi"
#                 name_patterns = [
#                     # Title + Name patterns
#                     r'^(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Prof\.?|Md\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
#                     # Standard name pattern (2-4 words, all starting with capital)
#                     r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$',
#                     # Name with comma (Last, First)
#                     r'^([A-Z][a-z]+,\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)$'
#                 ]
                
#                 for pattern in name_patterns:
#                     match = re.search(pattern, line)
#                     if match:
#                         name = match.group(1).strip()
#                         # Clean up the name
#                         name = re.sub(r'^[Mm]d\.?\s*', '', name)  # Remove "Md." prefix
#                         name = re.sub(r',\s*', ' ', name)  # Convert "Last, First" to "First Last"
#                         logger.info(f"Name extracted from content: {name}")
#                         return name
            
#             # Strategy 2: Look for "Resume" or "CV" followed by name
#             for i, line in enumerate(lines[:5]):
#                 if 'resume' in line.lower() or 'cv' in line.lower():
#                     # Check next line for potential name
#                     if i + 1 < len(lines):
#                         next_line = lines[i + 1].strip()
#                         if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', next_line):
#                             logger.info(f"Name extracted after resume header: {next_line}")
#                             return next_line
            
#             # Strategy 3: Use LLM to extract name if other methods fail
#             return self.extract_name_with_llm(content)
            
#         except Exception as e:
#             logger.error(f"Error extracting name from content: {e}")
#             return None
    
#     def extract_name_with_llm(self, content: str) -> Optional[str]:
#         """
#         Use LLM as fallback to extract name from content
#         """
#         try:
#             from langchain_google_genai import ChatGoogleGenerativeAI
#             from langchain.prompts import PromptTemplate
            
#             # Take first 1000 characters to avoid token limits
#             sample_content = content[:1000]
            
#             prompt_template = """
#             Extract the candidate's full name from this CV content. 
#             Return ONLY the name, nothing else.
            
#             CV CONTENT:
#             {content}
            
#             If you cannot find a name, return "NOT_FOUND".
            
#             CANDIDATE NAME:
#             """
            
#             llm = ChatGoogleGenerativeAI(
#                 model="gemini-2.5-flash-lite",
#                 temperature=0.1,
#                 convert_system_message_to_human=True
#             )
            
#             prompt = PromptTemplate(
#                 template=prompt_template,
#                 input_variables=["content"]
#             )
            
#             response = llm.invoke(prompt.invoke({"content": sample_content}))
#             name = response.content.strip()
            
#             if name and name != "NOT_FOUND" and len(name) > 3:
#                 logger.info(f"Name extracted with LLM: {name}")
#                 return name
#             else:
#                 return None
                
#         except Exception as e:
#             logger.warning(f"LLM name extraction failed: {e}")
#             return None
    
#     def get_best_candidate_name(self, filename: str, content: str) -> Dict[str, str]:
#         """
#         Get the best candidate name using multiple strategies
#         Returns: {"candidate_name": "Name", "name_source": "source"}
#         """
#         # Try content extraction first (most accurate)
#         content_name = self.extract_name_from_content(content)
#         if content_name and content_name != filename:
#             return {
#                 "candidate_name": content_name,
#                 "name_source": "content_extraction"
#             }
        
#         # Try filename extraction
#         filename_name = self.extract_name_from_filename(filename)
#         if filename_name and filename_name != filename:
#             return {
#                 "candidate_name": filename_name, 
#                 "name_source": "filename_extraction"
#             }
        
#         # Fallback to original filename
#         return {
#             "candidate_name": filename,
#             "name_source": "original_filename"
#         }
    
#     def load_hash_index(self) -> Dict[str, Any]:
#         """Load enhanced hash index with candidate data"""
#         if os.path.exists(self.hash_index_file):
#             try:
#                 with open(self.hash_index_file, "r") as f:
#                     data = json.load(f)
#                     if isinstance(data, dict) and "candidates" in data:
#                         return data
#                     else:
#                         return self._migrate_old_index(data)
#             except Exception as e:
#                 logger.error(f"Error loading hash index: {e}")
        
#         return {
#             "candidates": {},
#             "file_hashes": {}
#         }
    
#     def _migrate_old_index(self, old_index: Dict) -> Dict:
#         """Migrate from old hash index format to new candidate-centric format"""
#         logger.info("Migrating old hash index to new format")
#         new_index = {
#             "candidates": {},
#             "file_hashes": {}
#         }
        
#         for file_hash, filename in old_index.items():
#             if isinstance(filename, str):
#                 candidate_id = self.generate_candidate_id()
#                 candidate_name = self.extract_name_from_filename(filename)
                
#                 new_index["candidates"][candidate_id] = {
#                     "candidate_id": candidate_id,
#                     "candidate_name": candidate_name,
#                     "original_filename": filename,
#                     "upload_timestamp": datetime.now().isoformat(),
#                     "name_source": "filename_migration"
#                 }
#                 new_index["file_hashes"][file_hash] = candidate_id
        
#         logger.info(f"Migrated {len(new_index['candidates'])} candidates")
#         return new_index
    
#     def save_hash_index(self, hash_index: Dict[str, Any]):
#         """Save enhanced hash index"""
#         try:
#             with open(self.hash_index_file, "w") as f:
#                 json.dump(hash_index, f, indent=2, ensure_ascii=False)
#         except Exception as e:
#             logger.error(f"Error saving hash index: {e}")
#             raise
    
#     def register_candidate(self, file_hash: str, filename: str, content: str = None) -> Dict[str, str]:
#         """Register new candidate with enhanced name extraction"""
#         hash_index = self.load_hash_index()
        
#         # Check if file already processed
#         if file_hash in hash_index["file_hashes"]:
#             candidate_id = hash_index["file_hashes"][file_hash]
#             existing_candidate = hash_index["candidates"][candidate_id]
#             logger.info(f"File already processed for candidate: {existing_candidate['candidate_name']}")
#             return existing_candidate
        
#         # Get best candidate name using multiple strategies
#         if content:
#             name_data = self.get_best_candidate_name(filename, content)
#         else:
#             name_data = {
#                 "candidate_name": self.extract_name_from_filename(filename),
#                 "name_source": "filename_extraction"
#             }
        
#         # Create new candidate
#         candidate_id = self.generate_candidate_id()
        
#         candidate_data = {
#             "candidate_id": candidate_id,
#             "candidate_name": name_data["candidate_name"],
#             "original_filename": filename,
#             "upload_timestamp": datetime.now().isoformat(),
#             "name_source": name_data["name_source"]
#         }
        
#         # Update index
#         hash_index["candidates"][candidate_id] = candidate_data
#         hash_index["file_hashes"][file_hash] = candidate_id
        
#         self.save_hash_index(hash_index)
        
#         logger.info(f"Registered new candidate: {candidate_data['candidate_name']} ({candidate_id}) from {name_data['name_source']}")
#         return candidate_data
    
#     def get_candidate_by_id(self, candidate_id: str) -> Optional[Dict[str, Any]]:
#         """Get candidate data by ID"""
#         hash_index = self.load_hash_index()
#         return hash_index["candidates"].get(candidate_id)
    
#     def list_all_candidates(self) -> list:
#         """Get list of all candidates"""
#         hash_index = self.load_hash_index()
#         return list(hash_index["candidates"].values())


# candidate_manager.py
import uuid
import re
import os
from datetime import datetime
from typing import Dict, Any, Optional
import json

from utils.logger import logger

class CandidateManager:
    def __init__(self, hash_index_file: str):
        self.hash_index_file = hash_index_file
    
    def generate_candidate_id(self) -> str:
        """Generate unique UUID for candidate"""
        return str(uuid.uuid4())
    
    def extract_name_from_filename(self, filename: str) -> str:
        """
        Extract candidate name from filename
        """
        try:
            # Remove file extension
            name_only = os.path.splitext(filename)[0]
            
            # Remove common suffixes
            suffixes = ['_cv', '_resume', '-cv', '-resume', '_CV', '_Resume', ' cv', ' resume']
            for suffix in suffixes:
                name_only = name_only.replace(suffix, '')
            
            # Remove years and numbers
            name_only = re.sub(r'[_\-]\d{4}', '', name_only)
            name_only = re.sub(r'[_\-]\d+', '', name_only)
            
            # Replace separators with space
            name_only = re.sub(r'[_\-.]+', ' ', name_only)
            
            # Title case and clean up
            name_parts = [part.strip().title() for part in name_only.split() if part.strip()]
            candidate_name = ' '.join(name_parts)
            
            # Validate it looks like a name (at least 2 parts)
            if len(name_parts) >= 2 and all(len(part) > 1 for part in name_parts):
                logger.info(f"Name extracted from filename: {candidate_name}")
                return candidate_name
            else:
                logger.warning(f"Filename doesn't contain proper name: {filename}")
                return filename
                
        except Exception as e:
            logger.error(f"Error extracting name from filename {filename}: {e}")
            return filename
    
    def extract_name_from_content(self, content: str) -> Optional[str]:
        """
        Extract candidate name from CV content using multiple strategies
        """
        try:
            if not content:
                return None
            
            # Strategy 1: Look for name patterns at the beginning of document
            lines = content.split('\n')
            for i, line in enumerate(lines[:10]):  # Check first 10 lines
                line = line.strip()
                if not line:
                    continue
                
                # Common name patterns
                # "Md. Mahadi Hasan" or "Mahadi Hasan" or "Hasan, Mahadi"
                name_patterns = [
                    # Title + Name patterns
                    r'^(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Prof\.?|Md\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                    # Standard name pattern (2-4 words, all starting with capital)
                    r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$',
                    # Name with comma (Last, First)
                    r'^([A-Z][a-z]+,\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)$'
                ]
                
                for pattern in name_patterns:
                    match = re.search(pattern, line)
                    if match:
                        name = match.group(1).strip()
                        # Clean up the name
                        name = re.sub(r'^[Mm]d\.?\s*', '', name)  # Remove "Md." prefix
                        name = re.sub(r',\s*', ' ', name)  # Convert "Last, First" to "First Last"
                        logger.info(f"Name extracted from content: {name}")
                        return name
            
            # Strategy 2: Look for "Resume" or "CV" followed by name
            for i, line in enumerate(lines[:5]):
                if 'resume' in line.lower() or 'cv' in line.lower():
                    # Check next line for potential name
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', next_line):
                            logger.info(f"Name extracted after resume header: {next_line}")
                            return next_line
            
            # Strategy 3: Use LLM to extract name if other methods fail
            return self.extract_name_with_llm(content)
            
        except Exception as e:
            logger.error(f"Error extracting name from content: {e}")
            return None
    
    def extract_name_with_llm(self, content: str) -> Optional[str]:
        """
        Use LLM as fallback to extract name from content
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain.prompts import PromptTemplate
            
            # Take first 1000 characters to avoid token limits
            sample_content = content[:1000]
            
            prompt_template = """
            Extract the candidate's full name from this CV content. 
            Return ONLY the name, nothing else.
            
            CV CONTENT:
            {content}
            
            If you cannot find a name, return "NOT_FOUND".
            
            CANDIDATE NAME:
            """
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=0.1,
                convert_system_message_to_human=True
            )
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["content"]
            )
            
            response = llm.invoke(prompt.format_prompt(content=sample_content))
            name = response.content.strip()
            
            if name and name != "NOT_FOUND" and len(name) > 3:
                logger.info(f"Name extracted with LLM: {name}")
                return name
            else:
                return None
                
        except Exception as e:
            logger.warning(f"LLM name extraction failed: {e}")
            return None
    
    def get_best_candidate_name(self, filename: str, content: str) -> Dict[str, str]:
        """
        Get the best candidate name using multiple strategies
        Returns: {"candidate_name": "Name", "name_source": "source"}
        """
        # Try content extraction first (most accurate)
        content_name = self.extract_name_from_content(content)
        if content_name and content_name != filename:
            return {
                "candidate_name": content_name,
                "name_source": "content_extraction"
            }
        
        # Try filename extraction
        filename_name = self.extract_name_from_filename(filename)
        if filename_name and filename_name != filename:
            return {
                "candidate_name": filename_name, 
                "name_source": "filename_extraction"
            }
        
        # Fallback to original filename
        return {
            "candidate_name": filename,
            "name_source": "original_filename"
        }
    
    def load_hash_index(self) -> Dict[str, Any]:
        """Load enhanced hash index with candidate data"""
        if os.path.exists(self.hash_index_file):
            try:
                with open(self.hash_index_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "candidates" in data:
                        return data
                    else:
                        return self._migrate_old_index(data)
            except Exception as e:
                logger.error(f"Error loading hash index: {e}")
        
        return {
            "candidates": {},
            "file_hashes": {}
        }
    
    def _migrate_old_index(self, old_index: Dict) -> Dict:
        """Migrate from old hash index format to new candidate-centric format"""
        logger.info("Migrating old hash index to new format")
        new_index = {
            "candidates": {},
            "file_hashes": {}
        }
        
        for file_hash, filename in old_index.items():
            if isinstance(filename, str):
                candidate_id = self.generate_candidate_id()
                candidate_name = self.extract_name_from_filename(filename)
                
                new_index["candidates"][candidate_id] = {
                    "candidate_id": candidate_id,
                    "candidate_name": candidate_name,
                    "original_filename": filename,
                    "upload_timestamp": datetime.now().isoformat(),
                    "name_source": "filename_migration"
                }
                new_index["file_hashes"][file_hash] = candidate_id
        
        logger.info(f"Migrated {len(new_index['candidates'])} candidates")
        return new_index
    
    def save_hash_index(self, hash_index: Dict[str, Any]):
        """Save enhanced hash index"""
        try:
            with open(self.hash_index_file, "w") as f:
                json.dump(hash_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving hash index: {e}")
            raise
    
    def register_candidate(self, file_hash: str, filename: str, content: str = None, file_path: str = None) -> Dict[str, str]:
        """Register new candidate with enhanced name extraction and source path"""
        hash_index = self.load_hash_index()
        
        # Check if file already processed
        if file_hash in hash_index["file_hashes"]:
            candidate_id = hash_index["file_hashes"][file_hash]
            existing_candidate = hash_index["candidates"][candidate_id]
            logger.info(f"File already processed for candidate: {existing_candidate['candidate_name']}")
            return existing_candidate
        
        # Get best candidate name using multiple strategies
        if content:
            name_data = self.get_best_candidate_name(filename, content)
        else:
            name_data = {
                "candidate_name": self.extract_name_from_filename(filename),
                "name_source": "filename_extraction"
            }
        
        # Create new candidate with source path
        candidate_id = self.generate_candidate_id()
        
        candidate_data = {
            "candidate_id": candidate_id,
            "candidate_name": name_data["candidate_name"],
            "original_filename": filename,
            "upload_timestamp": datetime.now().isoformat(),
            "name_source": name_data["name_source"],
            "source_path": file_path  # NEW: Store the full source path
        }
        
        # Update index
        hash_index["candidates"][candidate_id] = candidate_data
        hash_index["file_hashes"][file_hash] = candidate_id
        
        self.save_hash_index(hash_index)
        
        logger.info(f"Registered new candidate: {candidate_data['candidate_name']} ({candidate_id}) from {name_data['name_source']}")
        return candidate_data
    
    def get_candidate_by_id(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Get candidate data by ID"""
        hash_index = self.load_hash_index()
        return hash_index["candidates"].get(candidate_id)
    
    def get_candidate_by_file_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get candidate data by file hash"""
        hash_index = self.load_hash_index()
        candidate_id = hash_index["file_hashes"].get(file_hash)
        if candidate_id:
            return hash_index["candidates"].get(candidate_id)
        return None
    
    def list_all_candidates(self) -> list:
        """Get list of all candidates"""
        hash_index = self.load_hash_index()
        return list(hash_index["candidates"].values())
    
    def get_source_paths_for_candidates(self, candidate_ids: list) -> list:
        """Get source paths for multiple candidate IDs"""
        hash_index = self.load_hash_index()
        source_paths = []
        
        for candidate_id in candidate_ids:
            candidate = hash_index["candidates"].get(candidate_id)
            if candidate and candidate.get("source_path"):
                source_paths.append(candidate["source_path"])
        
        return source_paths