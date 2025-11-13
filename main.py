from flask import Flask, render_template, request
import requests
import pickle, time, traceback
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

app = Flask(__name__)

# ---------------- Load ML Models ----------------
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

fake_model = load_model("models/fake_model.pkl")
fake_vectorizer = load_model("models/fake_vectorizer.pkl")
ai_model = load_model("models/ai_model.pkl")
ai_vectorizer = load_model("models/ai_vectorizer.pkl")

# ---------------- Selenium Driver ----------------
def make_driver(headless=False):
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("start-maximized")
    options.add_argument("--log-level=3")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

# ---------------- Fetch Reviews ----------------
def fetch_reviews_from_url(url, max_pages=3, headless=False):
    reviews = []
    driver = None
    try:
        driver = make_driver(headless=headless)
        driver.set_page_load_timeout(30)
        driver.get(url)
        time.sleep(2)
        for _ in range(3):
            driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(1)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        selectors = [
            "span[data-hook='review-body']",
            "div.t-ZTKy", "div._16PBlm", "div._2MImiq",
            "div.review-text", "div.rev_text", "div.review",
            "div.rvw", "div.user-review", "article.review",
            "div[itemprop='reviewBody']", "div[class*='review'] p",
            "p[class*='review']"
        ]
        for sel in selectors:
            for r in soup.select(sel):
                text = r.get_text(" ", strip=True)
                if text and len(text.split()) > 3:
                    reviews.append(text)

        if len(reviews) < 5:
            for p in soup.find_all("p"):
                text = p.get_text(" ", strip=True)
                if text and len(text.split()) > 6:
                    reviews.append(text)

        # Deduplicate
        clean = []
        seen = set()
        for r in reviews:
            rr = r.strip()
            if rr not in seen:
                seen.add(rr)
                clean.append(rr)
        return clean

    except Exception as e:
        print("❌ Selenium error:", e)
        traceback.print_exc()
        return []
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

# ---------------- Similarity ----------------
def review_matches_product(input_review, product_reviews, threshold=0.6):
    if not product_reviews:
        return False, 0.0, None
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([input_review] + product_reviews)
    sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    max_idx = sim_matrix.argmax()
    max_sim = float(sim_matrix[0, max_idx])
    return max_sim >= threshold, max_sim, product_reviews[max_idx]

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/features")
def features():
    return render_template("features.html")

@app.route("/check-review", methods=["GET","POST"])
def check_review():
    prediction = None
    show_ai = False
    similarity_info = ""
    product_url = ""

    if request.method == "POST":
        text = request.form.get("review", "").strip()
        product_url = request.form.get("product_url", "").strip()
        check_ai = request.form.get("check_ai")

        if not text:
            prediction = "⚠️ कृपया review लिखें।"
        else:
            fake_vec = fake_vectorizer.transform([text])
            fake_pred = fake_model.predict(fake_vec)[0]
            fake_label = "Fake" if fake_pred == 1 else "Real"

            ai_label = ""
            if check_ai:
                ai_vec = ai_vectorizer.transform([text])
                ai_pred = ai_model.predict(ai_vec)[0]
                ai_label = "AI-Generated" if ai_pred == 1 else "Human-Written"
                show_ai = True

            if product_url:
                product_reviews = fetch_reviews_from_url(product_url, max_pages=3, headless=False)
                if not product_reviews:
                    similarity_info = "⚠️ URL से reviews नहीं मिले — product review page try करें।"
                else:
                    matched, sim_score, most_similar = review_matches_product(text, product_reviews)
                    if matched:
                        similarity_info = f"✅ Review similar (similarity={sim_score:.2f}): \"{most_similar}\""
                    else:
                        similarity_info = f"⚠️ Review not similar (max similarity={sim_score:.2f}): \"{most_similar}\""

            prediction = fake_label
            if ai_label:
                prediction += f" ({ai_label})"

    return render_template("index.html",
                           prediction=prediction,
                           show_ai=show_ai,
                           similarity_info=similarity_info,
                           product_url=product_url)

# ---------------- Contact Form ----------------
GOOGLE_FORM_ACTION = "https://docs.google.com/forms/d/e/1FAIpQLScbkQo5m45eZtwTU80zeWENCj63gkO9TIHxUJtBD8tpJoyr5A/formResponse"
ENTRY_NAME = "entry.2005620554"
ENTRY_EMAIL = "entry.1045781291"
ENTRY_SUBJECT = "entry.839337160"
ENTRY_MESSAGE = "entry.1166974658"

@app.route("/contact")
def contact():
    return render_template("contact.html", success=False)

@app.route("/submit_contact", methods=["POST"])
def submit_contact():
    data = {
        ENTRY_NAME: request.form.get("name"),
        ENTRY_EMAIL: request.form.get("email"),
        ENTRY_SUBJECT: request.form.get("subject"),
        ENTRY_MESSAGE: request.form.get("message")
    }
    try:
        response = requests.post(GOOGLE_FORM_ACTION, data=data)
        return render_template("contact.html", success=True)
    except:
        return render_template("contact.html", success=False)

if __name__ == "__main__":
    app.run(debug=True)
