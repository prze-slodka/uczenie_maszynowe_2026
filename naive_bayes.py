import math as m
X = [("Kup telewizor 20% taniej", "spam"), ("czy piwo jutro?", "ham"),("promocje na warzywa!", "spam"), ("spotkanie o 18", "ham"), ("super okazja kup teraz", "spam"), ("idziemy na wykład?", "ham"),("co robisz wieczorem?", "ham"),("spotkajmy się jutro rano", "ham"),("pamiętaj o zakupach", "ham"),("wyślij mi notatki z zajęć", "ham"),("idziemy dziś na siłownię?", "ham"),("zadzwonię później", "ham"),("masz chwilę na rozmowę?", "ham"),("do zobaczenia na uczelni", "ham"),("czy masz czas jutro?", "ham"),("spotkanie zostało przełożone", "ham"),("wygraj iphone kliknij teraz", "spam"),("promocja tylko dzisiaj 50% taniej", "spam"),("zarób szybko pieniądze online", "spam"),("kup teraz zapłać mniej", "spam"),("oferta specjalna tylko dla ciebie", "spam"),("kliknij tutaj aby odebrać nagrodę", "spam"),("darmowa dostawa tylko dziś", "spam"),("super rabaty na elektronikę", "spam"),("okazja dnia nie przegap", "spam"),("kup taniej już teraz", "spam"),]

ogolne = []
spam_words = {}
ham_words = {}
spam_words_count = 0
ham_words_count = 0

for x in X:
    for word in x[0].split():
        ogolne.append(word)
        if x[1] == "spam":
            spam_words[word] = spam_words.get(word, 0) + 1
            spam_words_count += 1
        elif x[1] == "ham":
            ham_words[word] = ham_words.get(word, 0) + 1
            ham_words_count += 1

vocab = set(ogolne)
vocab_size = len(vocab)

test = "okazje zapłać mniej kup teraz"
test_words = test.split()
count_spam = 0
count_ham = 0

count_spam_messages = sum(1 for x in X if x[1]=="spam")
count_ham_messages = sum(1 for x in X if x[1]=="ham")
p_spam = count_spam_messages / len(X)
p_ham = count_ham_messages / len(X)

score_spam = m.log(p_spam)
score_ham = m.log(p_ham)

for word in test_words:
    p_word_spam = (spam_words.get(word, 0) + 1) / (spam_words_count + vocab_size)
    score_spam += m.log(p_word_spam)

    p_word_ham = (ham_words.get(word, 0) + 1) / (ham_words_count + vocab_size)
    score_ham += m.log(p_word_ham)

if score_spam > score_ham:
    print("SPAM!!!")
else:    
    print("git")