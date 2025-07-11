import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

spam_data = pd.DataFrame({
    'text': [
        # SPAM
        'Congratulations! You have won a $1000 Walmart gift card.',
        'URGENT! Your mobile number has won $5000.',
        'You have won a free lottery ticket. Claim now!',
        'Free entry in 2 a weekly competition to win FA Cup tickets.',
        'WINNER!! Click here to claim your prize.',
        'Get Viagra now at a 50% discount!',
        'You’ve been selected for a $1000 cash prize.',
        'Earn money from home with this simple trick.',
        'Get cheap meds now, no prescription needed.',
        'This is your last chance to win!',
        'Limited offer! Buy 1 get 1 free.',
        'Win a free vacation to Bahamas now!',
        'Claim your reward before it expires.',
        'You have 1 unread prize message.',
        'Exclusive deal for you. Click now.',
        'Lowest price on car insurance. Check now!',
        'You are selected for a free loan.',
        'Click here to get free crypto coins!',
        'Free gift card just for you!',
        'Winner of the week! Check rewards.',
        'Congratulations! You’ve won an iPhone.',
        'Act now to get $500 free cash.',
        'Special offer just for subscribers.',
        'Earn from your phone now!',
        'Free Netflix for a year! Claim it.',

        # HAM
        'Hey, are we still meeting tonight?',
        'Don’t forget the meeting at 3 PM.',
        'Can you send me the report by evening?',
        'Let’s catch up soon. Been a while!',
        'Call me when you get free.',
        'What time is the movie today?',
        'Did you get the groceries?',
        'Happy birthday! Have a great day!',
        'Let me know when you reach.',
        'I’ll be there in 10 minutes.',
        'Great job on the presentation today.',
        'Can you pick me up at 6?',
        'Dinner is at 8, don’t be late.',
        'I’m in class, text later.',
        'Thanks for your help earlier.',
        'How was your weekend?',
        'Just got back from the gym.',
        'Finished reading the book you gave me.',
        'I left my charger at your place.',
        'Good night, talk tomorrow.',
        'Wanna grab coffee tomorrow?',
        'Don’t forget to email John.',
        'See you at the wedding!',
        'Your package has been delivered.',
        'The weather is nice today.'
    ],
    'label': ['spam'] * 25 + ['ham'] * 25
})

X_test = spam_data['text']
y_spam = spam_data['label']

X_train, X_test, y_train, y_test = train_test_split(X_test, y_spam, test_size=0.2, random_state=42)


spam_pipeline = Pipeline(steps=[
    ('vectorizer', TfidfVectorizer()),
    ('classification', KNeighborsClassifier(n_neighbors=2))
])

spam_pipeline.fit(X_train, y_train)

prediction = spam_pipeline.predict(['Happy 4th! (A gift inside)'])

print(f'Accuracy = {prediction}')