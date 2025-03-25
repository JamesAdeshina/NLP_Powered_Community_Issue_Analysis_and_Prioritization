from transformers import pipeline

# Initialize the summarization pipeline and paraphrasing pipeline
summarizer = pipeline("summarization")
paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphraser")

# Example documents
documents = [
    "Dear Sir/Madam, I am writing to express my sincere appreciation for the recent improvements in road maintenance on High Street, Bristol. The newly resurfaced lanes and prompt pothole repairs have greatly enhanced my daily commute. Since the council began their work in February 2023, the roads have been noticeably smoother, and the enhanced signage has significantly improved safety for both drivers and pedestrians. I was particularly impressed by the swift response to minor issues and the regular updates provided by the maintenance teams. This initiative has not only improved road safety but also boosted community morale. I kindly request that the council continue with these high standards and extend similar efforts to other areas in our district. Thank you for your dedication to the community and for making our city a better place to live. Sincerely, John D.",
    "Dear Council Team, I wish to commend the outstanding work on the refurbishment of Oxford Road in Manchester. The recent efforts to repair cracks and repaint road markings have noticeably improved both the aesthetics and the safety conditions for local residents and commuters. The work carried out in March 2023 was efficient and thorough, leaving the area much cleaner and more navigable. It is heartening to see such dedication and attention to detail from the council’s maintenance crew. I believe these improvements are a positive step towards better public infrastructure, and I encourage similar projects throughout the city. Thank you for your commitment to enhancing our roads and the overall community environment. Yours sincerely, Sarah P.",
    "Dear Council, I am writing to express my deep concern regarding the consistently overflowing trash bins on Elm Street in Leeds. Over the past few months, the bins have rarely been emptied in a timely manner, resulting in unpleasant odors and attracting pests to the area. On several occasions, the situation has escalated to unsanitary conditions that affect the well-being of local residents. Despite previous reports submitted in November 2022, little has been done to address this issue. The lack of regular maintenance is a serious problem that undermines our quality of life. I urge the council to take immediate action by increasing waste collection frequency and providing additional bins if necessary. Thank you for your prompt attention to this urgent matter. Sincerely, Michael R."
]

# Combine all the documents into one large text
combined_text = " ".join(documents)

# Summarize the combined text
summary = summarizer(combined_text, max_length=300, min_length=100, do_sample=False)

# Convert the summary to a human-friendly tone
def make_human_friendly(summary_text):
    return (
        "Here's a summary of the documents:\n\n"
        "In these documents, we received a lot of valuable information covering various topics. "
        "The key points across all the documents are as follows:\n\n" +
        summary_text + "\n\n"
        "Thank you for taking the time to read through this summary. We hope this information provides you with a clear overview of the content."
    )

# Extract summary text from the summarizer output
final_summary = make_human_friendly(summary[0]['summary_text'])

# Paraphrase the human-friendly summary
paraphrased_summary = paraphraser(final_summary, max_length=500, num_return_sequences=1)

# Output the final paraphrased summary
print("Paraphrased Human-like Summary:", paraphrased_summary[0]['generated_text'])
