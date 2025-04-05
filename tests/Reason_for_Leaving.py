from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Example dataset with reasons for leaving a job (long text)
existing_text = [
    "After several years of working in the same role and contributing meaningfully to the organization, I began to feel that my professional growth had plateaued. I wanted to explore new opportunities that could challenge my current skill set, inspire innovation, and expand my expertise in different areas. Staying in my comfort zone no longer felt fulfilling, as I aimed to embrace fresh perspectives and collaborate with diverse teams. This decision was about pushing the boundaries of what I could achieve, both personally and professionally, while seeking a dynamic work environment where I could make an even greater impact.",

    "Due to personal circumstances, I needed to relocate to a new city to support my family and adjust to changes in my personal life. Although the relocation was an exciting opportunity to start fresh in a different environment, it also meant that continuing in my previous role became impractical. Balancing these personal commitments with my career goals was challenging, but I am now motivated to seek opportunities in the new city that align with my aspirations. I am eager to establish myself professionally while ensuring that my personal and family priorities remain in balance throughout this transition.",

    "I decided to pursue further education to not only deepen my expertise but also stay ahead in a fast-evolving job market. Continuous learning has always been a personal and professional priority for me, and I saw this as an ideal moment to enhance my qualifications. This decision was driven by a desire to strengthen my foundation in areas directly relevant to my career aspirations and to open doors to new opportunities. By investing in higher education, I aimed to equip myself with the skills and knowledge required to excel in an increasingly competitive and dynamic professional landscape.",

    "The company I was previously employed at underwent significant downsizing, which unexpectedly impacted my role. While this was a challenging experience, it provided me with an opportunity to reflect on my career trajectory and explore new possibilities. I took this as a chance to rethink how my skills, expertise, and experiences could be applied to other industries or roles. Although the transition was initially daunting, I’ve come to view it as a turning point—an opportunity to find an organization where my contributions will be valued, and where I can grow within a more stable and forward-thinking environment.",

    "After giving the matter considerable thought, I realized that my previous role was not providing the work-life balance I needed. The demands of the position made it challenging to dedicate time to personal priorities, which began to take a toll on my overall well-being. I made the decision to look for a position that would allow me to maintain high levels of productivity at work while still having time to invest in family, health, and personal interests. This transition marks a commitment to ensuring that my career aligns with my values, fostering both professional success and personal fulfillment.",

    "After many years of working diligently in my field, I made the decision to retire early and step away from the workforce. This decision was influenced by my desire to spend more time with family, pursue hobbies, and reflect on my achievements. However, after some time away, I realized that I still have the energy, skills, and passion to contribute meaningfully to the professional world. I now aim to explore opportunities where I can apply my expertise, mentor others, and remain active in a fulfilling role that enables me to strike a balance between professional engagement and personal enjoyment.",

    "Due to health-related concerns, I had to take some time off from work to focus on recovery and prioritize my well-being. This period was essential for me to regain stability, both physically and mentally, ensuring that I could return to the workforce with renewed energy and focus. Now that my condition has stabilized, I feel ready to re-enter a professional setting where I can bring my skills, experience, and dedication to the table. My goal is to find a position that supports a healthy work-life balance, allowing me to maintain my well-being while contributing meaningfully to the organization.",

    "I was seeking a transition to a part-time role to better accommodate other priorities in my life. Over time, I realized that the demands of my previous full-time position were no longer compatible with my goals of dedicating time to personal growth, family responsibilities, and other meaningful pursuits. By transitioning to part-time work, I aimed to maintain a strong connection to my professional field while also achieving greater flexibility. My objective is to find a role where I can contribute effectively, apply my expertise, and make a positive impact, all while enjoying the freedom to focus on personal aspirations."
]

# New applicant's reason for leaving
new_text = [
    "Having consistently excelled in my role, I decided to explore new opportunities that aligned more closely with my long-term career goals. While my previous position offered valuable experiences, I realized that I wanted to challenge myself further by working on larger-scale projects and collaborating with industry leaders. I was particularly motivated to find an organization where I could take on more leadership responsibilities, expand my skill set, and make a more significant impact. This transition represents my commitment to personal growth, professional development, and the pursuit of innovative work that inspires creativity and continuous learning."
]

# Step 1: Apply Lemmatization to Text
existing_text_lemmatized = [
    " ".join([lemmatizer.lemmatize(word) for word in sentence.split()]) for sentence in existing_text
]
new_text_lemmatized = [
    " ".join([lemmatizer.lemmatize(word) for word in sentence.split()]) for sentence in new_text
]

# Step 2: Vectorize the Lemmatized Textual Data (TF-IDF)
vectorizer = TfidfVectorizer()
existing_text_tfidf = vectorizer.fit_transform(existing_text_lemmatized).toarray()  # Text to numerical vectors
new_text_tfidf = vectorizer.transform(new_text_lemmatized).toarray()  # Transform new text

# Step 3: Fit K-Means Clustering Model
kmeans = KMeans(n_clusters=2, random_state=42)  # Two clusters for categorization
kmeans.fit(existing_text_tfidf)

# Step 4: Assign Labels to Clusters
cluster_labels = {0: "Career-Oriented Reasons", 1: "Personal or Circumstantial Reasons"}

# Step 5: Classify the New Text Data
predicted_cluster = kmeans.predict(new_text_tfidf)[0]  # Get the cluster number
category = cluster_labels[predicted_cluster]  # Map cluster number to label
print(f"This is under the category: '{category}'")

# Step 6: Visualize Clusters in 2D
pca = PCA(n_components=2)
existing_text_reduced = pca.fit_transform(existing_text_tfidf)
plt.scatter(existing_text_reduced[:, 0], existing_text_reduced[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
plt.title("Clusters of Reasons for Leaving")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")
plt.show()
