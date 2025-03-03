# course_recommender_app.py
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# --------------------------
# DATABASE SETUP
# --------------------------
engine = create_engine('sqlite:///users.db', connect_args={"check_same_thread": False})
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, default="user")  # "admin" or "user"

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
db_session = Session()

def create_user(username, password, role="user"):
    # Check if the username already exists
    if db_session.query(User).filter_by(username=username).first():
        return False
    new_user = User(username=username, password=password, role=role)
    db_session.add(new_user)
    db_session.commit()
    return True

def authenticate_user(username, password):
    return db_session.query(User).filter_by(username=username, password=password).first()

# --------------------------
# LOGIN & SIGNUP INTERFACE
# --------------------------
def show_login_signup():
    st.title("Welcome to Smart Course Recommender")
    auth_mode = st.selectbox("Choose Action", ["Login", "Sign Up"])

    if auth_mode == "Login":
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                user = authenticate_user(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username = user.username
                    st.session_state.user_role = user.role
                    st.success(f"Welcome back, {username}!")
                else:
                    st.error("Invalid username or password.")
    else:
        with st.form("signup_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                # Optionally, you could allow setting an admin role manually.
                if create_user(new_username, new_password):
                    st.success("Account created successfully. Please log in.")
                else:
                    st.error("Username already exists. Please choose another one.")

# --------------------------
# COURSE DATA & RECOMMENDER
# --------------------------
def load_and_preprocess():
    # Load datasets
    coursera = pd.read_csv('coursera.csv')
    udemy = pd.read_csv('udemy.csv')

    # Standardize Coursera data
    coursera = coursera.rename(columns={
        'Course Name': 'title',
        'Course Rating': 'rating',
        'Course Description': 'description',
        'Difficulty Level': 'level',
        'Course URL': 'url',
        'Skills': 'skills'
    })
    
    # Standardize Udemy data
    udemy = udemy.rename(columns={
        'course_name': 'title',
        'reviews_avg': 'rating',
        'course_description': 'description',
        'course url': 'url',
        'course_duration': 'duration',
        'students_count': 'enrollments'
    })

    # Create unique IDs
    coursera['course_id'] = ['C' + str(i) for i in range(1000, 1000 + len(coursera))]
    udemy['course_id'] = ['U' + str(i) for i in range(2000, 2000 + len(udemy))]

    # Combine datasets
    combined = pd.concat([
        coursera[['course_id', 'title', 'description', 'level', 'rating', 'url', 'skills']],
        udemy[['course_id', 'title', 'description', 'level', 'rating', 'url', 'duration', 'enrollments']]
    ], ignore_index=True)

    # Clean data
    combined = combined.dropna(subset=['title'])
    combined['title'] = combined['title'].astype(str).str.strip()
    combined['description'] = combined['description'].fillna('').str.lower()
    combined['rating'] = pd.to_numeric(combined['rating'], errors='coerce').fillna(3.5)
    combined['level'] = combined['level'].fillna('Intermediate').str.title()
    
    return combined

class HybridRecommender:
    def __init__(self, data):
        self.data = data
        self.cf_model = None
        self.tfidf_vectorizer = None
        self.cosine_sim = None
        self.user_ratings = self.generate_user_ratings()
    
    def generate_user_ratings(self):
        np.random.seed(42)
        num_users = 500
        num_interactions = 5000
        
        users = [f'user_{i}' for i in range(1, num_users+1)]
        courses = self.data['course_id'].sample(frac=0.5).tolist()
        
        return pd.DataFrame({
            'user_id': np.random.choice(users, num_interactions),
            'course_id': np.random.choice(courses, num_interactions),
            'rating': np.random.uniform(3, 5, num_interactions)
        })
    
    def train_collaborative_filtering(self):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.user_ratings, reader)
        trainset = data.build_full_trainset()
        
        self.cf_model = SVD(n_factors=20, n_epochs=25, lr_all=0.007, reg_all=0.1)
        self.cf_model.fit(trainset)
    
    def train_content_based(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.data['description'] + ' ' + self.data['level']
        )
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    def get_hybrid_recommendations(self, user_id, course_id, top_n=10):
        idx = self.data[self.data['course_id'] == course_id].index[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n*2]
        
        recommendations = []
        for i, score in sim_scores:
            course = self.data.iloc[i]
            try:
                pred_rating = self.cf_model.predict(user_id, course['course_id']).est
            except Exception:
                pred_rating = 3.5
                
            hybrid_score = (0.6 * pred_rating) + (0.4 * score)
            
            recommendations.append({
                'course_id': course['course_id'],
                'title': course['title'],
                'score': hybrid_score,
                'rating': course['rating'],
                'level': course['level'],
                'url': course['url'],
                'type': 'Coursera' if course['course_id'].startswith('C') else 'Udemy'
            })
        
        df = pd.DataFrame(recommendations).drop_duplicates('course_id')
        return df.sort_values('score', ascending=False).head(top_n)

# --------------------------
# MAIN APPLICATION
# --------------------------
def main():
    st.set_page_config(page_title="Smart Course Recommender", layout="wide", page_icon="🎓")
    
    # Initialize session state variables if not already set
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'user_role' not in st.session_state:
        st.session_state.user_role = "user"

    # Display login/signup if the user is not logged in.
    if not st.session_state.logged_in:
        show_login_signup()
        st.stop()

    # --------------------------
    # MAIN APP AFTER LOGIN
    # --------------------------
    st.sidebar.header("User Preferences")
    
    # For admin users, display a dropdown of available user IDs; for others, display their username.
    if st.session_state.user_role == "admin":
        users = db_session.query(User.username).all()
        user_list = [u[0] for u in users]
        user_id = st.sidebar.selectbox("Select Your User ID", user_list)
    else:
        st.sidebar.markdown(f"**Logged in as:** {st.session_state.username}")
        user_id = st.session_state.username

    search_term = st.sidebar.text_input("Search Courses:", "")

    # Load data and initialize recommender engine
    data = load_and_preprocess()
    recommender = HybridRecommender(data)
    
    with st.spinner('Initializing recommendation engine...'):
        recommender.train_content_based()
        recommender.train_collaborative_filtering()
    
    try:
        if search_term:
            filtered_courses = data[
                data['title'].str.contains(search_term, case=False, na=False)
            ]
            if filtered_courses.empty:
                st.warning("No courses found matching your search. Try different keywords.")
                return
        else:
            filtered_courses = data.sample(10)
            
        selected_course = st.sidebar.selectbox("Select a Course:", filtered_courses['title'], index=0)
    except Exception as e:
        st.error(f"Error processing search: {str(e)}")
        return

    st.title("🎓 Smart Course Recommendation Engine")
    st.markdown("Discover personalized learning recommendations across Coursera and Udemy!")
    
    if st.sidebar.button("Get Recommendations"):
        try:
            course_id = data[data['title'] == selected_course]['course_id'].values[0]
            with st.spinner(f'Finding recommendations similar to {selected_course}...'):
                recommendations = recommender.get_hybrid_recommendations(user_id, course_id)
            
            st.subheader("🔍 Recommended Courses")
            st.markdown(f"Based on **{selected_course}** and user **{user_id}**'s preferences")
            
            cols = st.columns(2)
            for idx, row in recommendations.iterrows():
                with cols[idx % 2]:
                    with st.container():
                        st.markdown(f"### [{row['title']}]({row['url']})")
                        st.markdown(f"**Platform:** {row['type']} | **Level:** {row['level']}")
                        st.markdown(f"**Recommendation Score:** {row['score']:.2f} | **Course Rating:** {row['rating']:.1f}/5")
                        if row['type'] == 'Udemy':
                            duration = data[data['course_id'] == row['course_id']]['duration'].values[0]
                            st.markdown(f"⏳ Duration: {duration}")
                        else:
                            skills = data[data['course_id'] == row['course_id']]['skills'].values[0]
                            st.markdown(f"🛠 Skills: {skills[:100]}...")
                        st.markdown("---")
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
    
    #Logout
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.user_role = "user"
        st.info("You have been logged out. Please refresh the page.")
        st.stop()

if __name__ == '__main__':
    main()