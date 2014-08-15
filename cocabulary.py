# -*- coding: utf-8 -*-
import pickle
from PCV.imagesearch import vocabulary
from PCV.tools.imtools import get_imlist
from PCV.localdescriptors import sift

imlist = get_imlist('./first500/')
nbr_images = len(imlist)
featlist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]

for i in range(nbr_images):
    sift.process_image(imlist[i], featlist[i])

voc = vocabulary.Vocabulary('ukbench')
voc.train(featlist, 1000, 10)
# saving vocabulary
with open('./first500/vocabulary.pkl', 'wb') as f:
    pickle.dump(voc, f)
print 'vocabulary is:', voc.name, voc.nbr_words