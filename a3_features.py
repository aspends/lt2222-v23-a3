import os
import sys
import argparse
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

def read_documents(inputdir):
    texts = []
    labels = []
    for author in os.listdir(inputdir):
        author_dir = os.path.join(inputdir, author)
        if os.path.isdir(author_dir):
            for filename in os.listdir(author_dir):
                filepath = os.path.join(author_dir, filename)
                if os.path.isfile(filepath):
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                        # Strip email headers and signatures
                        text = re.sub(r"^.*\nFrom:.*\n", "", text, flags=re.MULTILINE)
                        text = re.sub(r"\n-{2,}.*\n.*$", "", text, flags=re.MULTILINE)
                        text=re.sub(r'Message_ID:.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Date:.*\n','',text,flags=re.MULTILINE)
                        text = re.sub(r'From:.*\n', '', text,flags=re.MULTILINE)
                        text=re.sub(r'To:.*\n','',text,flags=re.MULTILINE)
                        text = re.sub(r'Subject:.*\n', '', text,flags=re.MULTILINE)
                        text=re.sub(r'Mime-Version:.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Content-Type:.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Content-Transfer-Encoding:.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'X-From:.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'X-To:.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'X-cc:.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'X-bcc:.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'X-Folder:.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'X-Origin:.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'X-FileName:.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r' -----Original Message-----.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Sent:.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'To:.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Cc:.*\n','',text,flags=re.MULTILINE)
                        #Sign offs
                        text=re.sub(r'Cordially,.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Susan S. Bailey.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Enron North America Corp..*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'1400 Smith Street, Suite 3803A.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Houston, Texas 77002.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Phone: (713) 853-4737.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Fax: (713) 646-3490.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Email: Susan.Bailey@enron.com.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Sent from my BlackBerry Wireless Handheld (www.BlackBerry.net).*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Sincerely.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Sincerely,.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Christi Nicolay.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'{contact info}.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Regards,.*\n','',text,flags=re.MULTILINE)
                        text = re.sub(r'[_-]{2,}.*\n', '', text,flags=re.MULTILINE)
                        text=re.sub(r'Craig.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Thanks!.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Thanks,.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Thank you.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Craig Dean.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Tom Donohoe.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Mr. Dean.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Dean.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Hope all is well,.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'skullman@netzero.net.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'nightflight.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'4702 Gladesdale Park Lane.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Katy, TX 77450  .*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'3-7151.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'713-853-7151.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Deano.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Thanks!.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Thanks..*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Rosie.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Eric.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Stephanie.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Senior Legal Specialist.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Enron Wholesale Services.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'1400 Smith Street, EB3803C.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Houston, Texas  77002.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'ph:  713.345.3249.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'fax:  713.646.3490.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'email:  stephanie.panus@enron.com.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'E.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Larry.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Larry May.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Lawrence J. May.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r' wk 713-853-6731 larry.may@enron.com.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'hm 281-379-1525 ljnmay@aol.com.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Lawrence May.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'713 853-6731 email: ljnmay@aol.com.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'14542 Kentley Orchard Ln..*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Cypress, Texas 77429.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Home: 281 379-1525  email: ljnmay@aol.com.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Work: 713 853-6731  email: larry.may@enron.com.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'713 853-6731.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'work: larry.may@enron.com.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'home: ljnmay@aol.com.*\n','',text, flags=re.MULTILINE)
                        text=re.sub(r'Ken.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Rosalee Fleming.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Ken Lay.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'3-6731.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Assistant to Ken Lay.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'<Embedded Picture (Metafile)>.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'_________________________________________________________________.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Chat with friends online, try MSN Messenger: http://messenger.msn.com..*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'Call me with any question,.*\n','',text, flags=re.MULTILINE)
                        text=re.sub(r'Clint.*\n','',text,flags=re.MULTILINE)
                        text=re.sub(r'C Dean.*\n','',text,flags=re.MULTILINE)
                        texts.append(text)
                        labels.append(author)
    return texts, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("vectorfile", type=str, help="The name of the output file containing the vector representations.")
    parser.add_argument("labelfile", type=str, help="The name of the output file containing the author labels.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default=20, help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    texts, labels = read_documents(args.inputdir)

    print("Processing {} documents...".format(len(texts)))
    vectorizer = CountVectorizer(stop_words="english", max_features=10000)
    X = vectorizer.fit_transform(texts)

    print("Reducing dimensionality to {}...".format(args.dims))
    svd = TruncatedSVD(n_components=args.dims, random_state=42)
    X_reduced = svd.fit_transform(X)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, labels, test_size=args.testsize/100, random_state=42)
    train_data = pd.DataFrame(X_train)
    train_data["label"] = y_train
    test_data = pd.DataFrame(X_test)
    test_data["label"] = y_test
    all_data = pd.concat([train_data, test_data])

    print("Writing to {}...".format(args.vectorfile))
    all_data.to_csv(args.vectorfile, columns=list(range(args.dims)), header=False, index=False)

    print("Writing to {}...".format(args.labelfile))
    label_data = pd.DataFrame({"label": labels})
    label_data.to_csv(args.labelfile, index=False)

    print("Done!")
