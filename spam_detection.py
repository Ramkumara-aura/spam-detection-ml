"""
Spam Message Detection using Machine Learning
------------------------------------------------
Trains and compares Naive Bayes, Logistic Regression, and Linear SVM
classifiers to detect spam vs. ham (legitimate) text messages using
TF-IDF features.

Author: Ram Kumar A

Usage:
    python spam_detection.py            # train + evaluate + save results
    python spam_detection.py --predict "Congratulations! You won a free prize"
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

RANDOM_SEED = 42
OUTPUT_DIR = "outputs"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------
SPAM_TEMPLATES = [
    "Congratulations! You have won a {prize} worth ${amount}. Claim now by clicking {link}",
    "URGENT: Your account has been suspended. Verify your details at {link} immediately",
    "You are selected for a free {prize}! Reply YES to claim your reward now",
    "Limited time offer! Get {amount}% discount on all products, click {link} today",
    "Dear customer, you have an unclaimed prize of ${amount}. Call now to claim",
    "WINNER!! As a valued customer you have been selected to receive a {prize}",
    "Your loan of ${amount} has been approved. Click {link} to receive the funds",
    "FREE entry into our {amount} lottery, just text WIN to enter now",
    "Act now! Your {prize} is waiting, offer expires today, visit {link}",
    "Hot singles in your area want to chat with you now, click {link}",
    "Your bank account needs verification, login at {link} to avoid suspension",
    "Cheap meds available online, no prescription needed, order at {link}",
    "You have been chosen to test our new {prize} for free, click here {link}",
    "Final notice: pay ${amount} now or your service will be disconnected",
    "Claim your free {prize} gift card now before it expires, {link}",
]

HAM_TEMPLATES = [
    "Hey, are we still meeting for lunch tomorrow at {time}?",
    "Can you send me the notes from today's class?",
    "Don't forget to submit the assignment by {time}",
    "I'll be at the office by {time}, see you then",
    "Thanks for helping me with the project yesterday",
    "Let's catch up this weekend, are you free?",
    "The meeting has been rescheduled to {time}",
    "Please review the document I shared and share your feedback",
    "Happy birthday! Hope you have a wonderful day",
    "Can we discuss the internship report tomorrow?",
    "I am running late, will reach by {time}",
    "Great job on the presentation today, well done",
    "Reminder: college fest starts on Monday",
    "Could you share the reference book for the exam?",
    "Let's plan the weekend trip, call me when free",
]

PRIZES = ["iPhone 15", "laptop", "gift voucher", "smartwatch", "vacation package", "cash prize"]
AMOUNTS = ["500", "1000", "50", "2000", "75", "10000"]
LINKS = ["bit.ly/claim-now", "secure-verify.net", "www.freegift.com", "shorturl.at/reward"]
TIMES = ["6 PM", "10 AM", "3 PM", "tomorrow morning", "5:30 PM", "noon"]

BORDERLINE_EXAMPLES = [
    ("Reminder: your subscription renews tomorrow, click here to manage", "spam"),
    ("Congratulations on completing the course, certificate attached", "ham"),
    ("Your order has been shipped, track it at bit.ly/claim-now", "ham"),
    ("Free pizza for all students at the fest today, come early", "ham"),
    ("Your package delivery failed, reschedule at secure-verify.net", "spam"),
    ("Win a free coffee coupon by attending today's seminar", "ham"),
]


def _fill(template: str) -> str:
    return template.format(
        prize=random.choice(PRIZES),
        amount=random.choice(AMOUNTS),
        link=random.choice(LINKS),
        time=random.choice(TIMES),
    )


def build_dataset(n_per_class: int = 150):
    """Builds a synthetic but realistic labeled SMS/email spam-vs-ham dataset."""
    data, labels = [], []
    for _ in range(n_per_class):
        data.append(_fill(random.choice(SPAM_TEMPLATES)))
        labels.append("spam")
    for _ in range(n_per_class):
        data.append(_fill(random.choice(HAM_TEMPLATES)))
        labels.append("ham")

    for text, label in BORDERLINE_EXAMPLES:
        data.append(text)
        labels.append(label)

    # Flip a small fraction of labels to simulate real-world noisy/mislabeled data
    noise_idx = random.sample(range(len(data)), k=int(0.04 * len(data)))
    for i in noise_idx:
        labels[i] = "ham" if labels[i] == "spam" else "spam"

    return data, labels


# ---------------------------------------------------------------------
# 2. Train & evaluate
# ---------------------------------------------------------------------
def train_and_evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data, labels = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.25, random_state=RANDOM_SEED, stratify=labels
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
    }

    results = {}
    best_name, best_acc, best_preds, best_model = None, -1, None, None

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, pos_label="spam"),
            "recall": recall_score(y_test, preds, pos_label="spam"),
            "f1": f1_score(y_test, preds, pos_label="spam"),
        }

        if results[name]["accuracy"] > best_acc:
            best_acc = results[name]["accuracy"]
            best_name, best_preds, best_model = name, preds, model

    print("Results per model:")
    for name, m in results.items():
        print(
            f"  {name:<20} acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
            f"rec={m['recall']:.3f}  f1={m['f1']:.3f}"
        )
    print(f"\nBest model: {best_name} ({best_acc:.1%} accuracy)")

    _save_confusion_matrix(y_test, best_preds, best_name)
    _save_model_comparison(results)
    _save_results_json(results, best_name, best_acc)

    return vectorizer, best_model, best_name


def _save_confusion_matrix(y_test, best_preds, best_name):
    cm = confusion_matrix(y_test, best_preds, labels=["spam", "ham"])

    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=200)
    ax.imshow(cm, cmap="Purples")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Spam", "Ham"], fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Spam", "Ham"], fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("Actual Label", fontsize=11, labelpad=8)
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=12, fontweight="bold", pad=12)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color=color, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), facecolor="white")
    plt.close()


def _save_model_comparison(results):
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=200)
    names = list(results.keys())
    accs = [results[n]["accuracy"] * 100 for n in names]
    bars = ax.bar(names, accs, color=["#8B5CF6", "#A78BFA", "#C4B5FD"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Model Accuracy Comparison", fontsize=12, fontweight="bold")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 2, f"{a:.1f}%",
                 ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), facecolor="white")
    plt.close()


def _save_results_json(results, best_name, best_acc):
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump({"results": results, "best_model": best_name, "best_accuracy": best_acc}, f, indent=2)


# ---------------------------------------------------------------------
# 3. Predict a single message (CLI demo)
# ---------------------------------------------------------------------
def predict_message(message: str, vectorizer, model) -> str:
    vec = vectorizer.transform([message])
    return model.predict(vec)[0]


def main():
    parser = argparse.ArgumentParser(description="Spam Message Detection using Machine Learning")
    parser.add_argument("--predict", type=str, default=None,
                         help="Classify a single message as spam or ham")
    args = parser.parse_args()

    vectorizer, model, best_name = train_and_evaluate()

    if args.predict:
        label = predict_message(args.predict, vectorizer, model)
        print(f'\n[{best_name}] "{args.predict}" -> {label.upper()}')


if __name__ == "__main__":
    main()
