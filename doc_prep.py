from glob import glob
import pickle

def prepare_document(file_name):
    with open(file_name, encoding='utf-8') as f:
        doc = f.readlines()[2:]
        doc = ''.join(doc)
        doc = doc.replace('\u3000', '')
        doc = doc.replace('\n', '')
    return doc

def preprocess(doc_name, label_name):
    folders = glob('text/*')
    documents, labels = [], []
    i=0
    print('start processing')

    for (label, folder) in enumerate(folders):
        file_path = glob('{}/*'.format(folder))

        for file_name in file_path:
            document = prepare_document(file_name)
            documents.append(document)
            labels.append(label)
            i+=1
            if i%1000==0:
                print('processing: '+ str(i))


    doc_file = open(doc_name, 'wb')
    pickle.dump(documents, doc_file)
    labels_file = open(label_name, 'wb')
    pickle.dump(labels, labels_file)
    print('processing completed')
          
    return

if __name__ == "__main__":
    import sys
    preprocess(sys.argv[1], sys.argv[2])


