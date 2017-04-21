import matplotlib.pyplot as plt #çizim için gerekli kütüphane
from sklearn.datasets import load_digits #makine öğrenmesi kütüphanesinden gerekli verileri aldık

digits = load_digits() #Rakamların olduğu resimleri yüklüyoruz.

import pylab as pl #çizim için gerekli kütüphane

images_and_labels = list(zip(digits.images , digits.target))
plt.figure(figsize=(5,5))
#Çizdirme işlemleri aşağıdaki döngüde yapılacaktır
for index, (image ,label) in enumerate(images_and_labels[:15]):
    plt.subplot(3,5,index+1)
    plt.axis("off")
    plt.imshow(image , cmap=plt.cm.gray_r , interpolation="nearest")
    plt.title('%i' % label)

import random
from sklearn import ensemble

#Değişkenler tanımlandı
n_samples = len(digits.images)
x = digits.images.reshape((n_samples, -1))
y = digits.target

#Rastgele indisler seçildi
sample_index=random.sample(range(len(x)),len(x)/5) #20-80
valid_index=[i for i in range(len(x)) if i not in sample_index]

#Sample ve validation için resimler
sample_images=[x[i] for i in sample_index]
valid_images=[x[i] for i in valid_index]

#Sample ve validation hedefleri
sample_target=[y[i] for i in sample_index]
valid_target=[y[i] for i in valid_index]

#Random Tree Classifier algoritmasını kullanarak
classifier = ensemble.RandomForestClassifier()

classifier.fit(sample_images, sample_target)

#Validation verisini tahmin et
score=classifier.score(valid_images, valid_target)
print 'Random Tree Classifier:\n' 
print 'Skor:\t'+str(score)

i=150 # maksimum : 1797 <- örnek sayısı

pl.gray() 
pl.matshow(digits.images[i]) 
pl.show() 
result = classifier.predict(x[i])

print "Tahmin sonucu  : " , result