import os
import re
import webvtt
from llama_index.core import Document

def parse_vtt_to_documents(data_dir="data"):
    documents = []
    
    # Walk through the data directory to find all .vtt files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.vtt'):
                file_path = os.path.join(root, file)
                
                # Extract course name and session based on folder structure
                path_parts = file_path.split(os.sep)
                course_name = path_parts[-2]
                session_name = file.replace('.vtt', '')
                
                vtt = webvtt.read(file_path)
                
                chunk_text = ""
                chunk_start_time = None
                caption_count = 0
                chunk_speakers = set() # Track unique speakers in this 15-caption chunk
                
                for caption in vtt:
                    if chunk_start_time is None:
                        chunk_start_time = caption.start_in_seconds
                        
                    clean_text = caption.text.replace('\n', ' ').strip()
                    
                    # Extract speaker name if present (e.g., "Dr. Karan Mittal: ...")
                    # We look for text before a colon, capping length to avoid false positives
                    speaker_match = re.match(r"^([^:]+):\s*(.*)", clean_text)
                    if speaker_match:
                        speaker_name = speaker_match.group(1).strip()
                        if len(speaker_name) < 40: # Sanity check
                            chunk_speakers.add(speaker_name)
                    
                    # We leave the speaker name IN the text so the LLM knows who is talking
                    chunk_text += clean_text + " "
                    caption_count += 1
                    
                    if caption_count >= 15:
                        doc = Document(
                            text=chunk_text.strip(),
                            metadata={
                                "course_id": course_name,
                                "session": session_name,
                                "start_time_seconds": int(chunk_start_time),
                                "video_url": f"https://vidur.co/courses/view/{course_name}/{session_name}",
                                "speakers": list(chunk_speakers) # Save speakers as a list
                            }
                        )
                        documents.append(doc)
                        
                        # Reset for the next chunk
                        chunk_text = ""
                        chunk_start_time = None
                        caption_count = 0
                        chunk_speakers = set()
                        
                # Add any remaining text at the end of the file
                if chunk_text.strip():
                    doc = Document(
                        text=chunk_text.strip(),
                        metadata={
                            "course_id": course_name,
                            "session": session_name,
                            "start_time_seconds": int(chunk_start_time),
                            "video_url": f"https://vidur.co/courses/{course_name}/{session_name}",
                            "speakers": list(chunk_speakers)
                        }
                    )
                    documents.append(doc)
                    
    return documents

if __name__ == "__main__":
    print("Parsing VTT files...")
    # Make sure this path correctly points to your data folder from the src folder
    docs = parse_vtt_to_documents("../data") 
    
    print(f"Total context chunks created: {len(docs)}")
    
    if docs:
        print("\n--- Sample Document Metadata ---")
        print(docs[0].metadata)
        print("\n--- Sample Document Text ---")
        print(docs[0].text[:300] + "...")