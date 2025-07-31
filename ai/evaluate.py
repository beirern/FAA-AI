from app.query import full_query

print("\n--- Querying Examples ---")

queries = [
    "What kind of engine does the Cessna 162 use?",
    "Describe the electrical system of the Cessna 172.",
    "What are the rules for VFR flight?",
    "According to the Airman Certification Standards for the Private Pilots license what skills when maneuvering during slow flight do I need to exhibit and what are their codes?",
    "Summarize Chapter 1 Introduction to Flying for me from the PHAK.",
    "According to the AFH what is the protocol for Turbulent Air Approach and Landing?",
    "You are an FAA exam preparation assistant. Generate one multiple-choice question with four possible answers about the engine of a Cessna 162 based on the POH of the 162. The question should test a key concept from the text. Clearly indicate the correct answer."
]

for i, query in enumerate(queries):
    print(f"\n--- Query {i+1}: {query} ---")
    response = full_query(query)
    print(response)
    print("\nSource Document(s) and Page Number(s):")
    
    # Iterate through the source_nodes to get file name and page label
    # Each node in response.source_nodes is a NodeWithScore object
    for node_with_score in response.source_nodes:
        # Access the underlying TextNode (or other Node type)
        node = node_with_score.node
        
        # Get metadata from the node
        file_name = node.metadata.get('file_name', 'N/A')
        page_label = node.metadata.get('page_label', 'N/A') # 'page_label' is the common key for page number
        
        print(f"- File: {file_name}, Page: {page_label}")
        # Optionally, you can print a snippet of the content to verify
        # print(f"  Content Snippet: {node.get_content()[:150]}...")query_llm()