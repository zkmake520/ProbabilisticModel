#here we can use wrapper to accerlate the whole process, since many text may be same, we can save the intermediate results
import operator
from math import log10
import re
import string
import random
import heapq
"""First Part:
        Word Segmentation"""
def memory(f):
	#memorize function f
    table = {}
    def fmemo(*args):
        if args not in table:
            table[args] = f(*args)
        return table[args]
    return fmemo

#this memory procee is really important which makes the time from 2^n ->n^2*L
@memory
def Segment(text):
	#return a list of words that is the best segmentation of the text"
	#recursive implementation 
	if not text: return []
	candidates = ([first]+Segment(remind) for (first,remind) in Split(text))

	#TODO: actually we can store the Probabilty of each best Segment. there is no need to compute it again
	return max(candidates,key=bPwords) #key specifies a one-argument ordering function 
						

#L parameter is 20 in default ,this method returns a list of all possible (frist,rem) pairs, len(first)<=L
def Split(text,L=20):
	return [(text[:i+1],text[i+1:]) for i in range(min(len(text),L))]
# The Naive Bayes probabilities of a sequence of words
def Pwords(words):
	return Product([_P(word) for word in words])


def Product(nums):
	return reduce(operator.mul,nums)

#P(word) == count(word)/N since the GOOGLE N is corpus size. and note that nearly most common 1/3 of a million words covers 98% of all tokens
#so we can use only this part of words, and we can eliminate those numbers and punctuations.
def constantWord(word,N):
	return 1./N
def avoidLongWord(word,N):
	return 10./(N*10**len(word))

class Pdict(dict): 
	#probability distribution of words estimated from the counts in datafile
	def __init__(self,data,N=None,missing=constantWord):
		for key,count in data:
			self[key] = self.get(key,0) + int(count)
		self.N = float(N or sum(self.itervalues()))
		self.missing = missing

	def __call__(self,key):
		if key in self: return float(self[key]/self.N)
		else: return self.missing(key,self.N)

def Datafile(name,sep='\t'):
	for line in file(name):
		yield line.split(sep)


_N = 1024908267229	#Number of tokens
_P = Pdict(Datafile('vocab.txt'),_N,avoidLongWord)

###biagram
##model P(W1:n) = TTk=1:nP(Wk|Wk-1)
def Pwords2(word,pre):
	words = pre+' '+word
	if words not in _P2:
		return _P(word)
	else: return _P2(pre+' '+word)/float(_P(pre))

_P2 = Pdict(Datafile('count_2w.txt'),_N)

@memory
def Segment2(text,pre="<S>"):
	#return (log(P(words)),words) where words is the best segment
	if not text: return (0.0,[])
	candidates= [combine(log10(Pwords2(first,pre)),first,Segment2(remind,first)) for first,remind in Split(text)]
	return max(candidates)

def combine(Pfirst, first, (Prem,rem)):
	return (Pfirst+Prem,[first]+rem)

"""Second Part:
		Secret Code"""



alphabet = 'abcdefghijklmnopqrstuvwxyz'
def Encode(msg,key):
	#encode string with the substitution key
	return msg.translate(string.maketrans(ul(alphabet),ul(key)))

def ul(text): return text.upper()+text.lower()

def Encode_Shift(msg,n=10):
	#encode string with a shift(caesar) cipher
	return Encode(msg,alphabet[n:]+alphabet[:n])

#we can use the technique as above use a logPwords to decode without knowing the key
def logPwords(words):
	if isinstance(words,str): words=getAllWords(words)
	return sum(log10(_P(word)) for word in words)

def getAllWords(words):
	#return a list of words in string lowercase,use pattern compare
	return re.findall("[a-z]+",words.lower())

def Decode_Shift(msg):
	candidates = [Encode_Shift(msg,n) for n in range(len(alphabet))]
	return max(candidates,key=logPwords)
#note that above way is too easy 
#here we want to substitute using a general cipher,in which any letter can be substitued for any other letter
"""
     step1: given a encoded msg,split them into lowercase only words, and combine these words(remove those numbers and punctuations)
	 step2: from a random x, use local search to get to the local minimum, design the cost function
	 step3: repeat step2

"""
#use letter n-grams model
P3l = Pdict(Datafile("count_3l.txt"))
P2l = Pdict(Datafile("count_2l.txt"))


def localsearch(x,f,neighbors,steps=10000):
	#local search to get a x that maximizes function cost f
	fx = f(x)
	
	neighborhood = iter(neighbors(x))
	for i in range(steps):
		#print i,fx
		x2 = neighborhood.next()
		fx2 = f(x2)
		if fx2 > fx:
			x,fx = x2,fx2
			neighborhood = iter(neighbors(x))
	print x 
	return x

_cat ="".join

def Shuffle(text):
	text = list(text)
	random.shuffle(text)
	return text


def DecodeGeneral(msg,step=4000,restarts=20):
	#decode a general cipher string by using local search
	msg = cat(getAllWords(msg))   #just keep words of alphabet,lowercase
	print msg
	candidates= [localsearch(Encode(msg,key=cat(Shuffle(alphabet))),logP3letters,getNeighbors,step) for i in range(restarts)]
	(p,words) = max(Segment2(text) for text in candidates)
	return ' '.join(words)

def getNeighbors(msg):
	#generate nearby strings
	def swap(a,b):
		return msg.translate(string.maketrans(a+b,b+a))
	for bigram in heapq.nsmallest(20,set(ngrams(msg,2)),P2l):
		print bigram
		b1,b2=bigram
		for c in alphabet:
			if b1 == b2:
				if P2l(c+c) > P2l(bigram): yield swap(c,b1)
			else: 
				if P2l(c+b2) > P2l(bigram): yield swap(c,b1) 
				if P2l(b1+c) > P2l(bigram): yield swap(c,b2) 
	while True: 
		yield swap(random.choice(alphabet), random.choice(alphabet)) 
cat = ''.join 
"""
	Spelling Correction:
		Find argmaxcP(c|w) which means type w, c is the candidates find highest probability of c
		use bayes rule P(c|w) = P(w|c) +P(c)
		P(c) is straightforward
		P(w|c) is called error model,we need more data in http://www.dcs.bbk.ac.uk/~ROGER/corpora.html. 
		the data is not large enough, we can hope to just look up P(w=thaw|c=thew), changes are slim
		we do some trick by ignoring the letters that are same, then we get P(w=a|c=e) the probability that a was typed 
		when the corrector is e
""" 
def AllCorrections(text):
	#spelling correction for all words in text
	return re.sub('[a-zA-Z]+',lambda match:getCorrect(match.group(0)),text)

def getCorrect(word):
	#return word that is most likely to be the correct spelling of word
	candidates = getEdits(word).items()
	c,edit  = max(candidates, key=lambda (c,e):Pedit(e)*_P(c))
	return c

_Pe=Pdict(Datafile('count_1edit.txt'))
_PSpellError = 1./20
def Pedit(edit):
	#the probability of an edit,can be "" or 'a|b' or 'a|b + c|d'
	if edit == "": return (1.- _PSpellError)
	return _PSpellError*Product(_Pe(e) for e in edit.split("+"))

_Prefix = set(w[:i] for w in _P for i in range(len(w)+1))
#we can optimize it ,we don't need to consider all the edits, since merely of them are in vacabulary,thus we can precomputing all the 
#possible prefixes,and split the word into two parts, head and tail, thus the head should be in the prefix
def getEdits(w,dis=2):

	#return a dict of {correct:edit} pairs within dis edits of words
	res = {}
	def editsR(head,tail,d,edits):
		def edit(L,R): return edits+[R+'|'+L]
		C = head+tail
		if C in _P:
			e = '+'.join(edits)
			if C not in res: res[C] = e
			else: res[C] = max(res[C],e,key=Pedit)
		if d<=0: return 
		extensions = [head+c for c in alphabet if head+c in _Prefix] ##given a head, all possible heads
		pre = (head[-1] if head else '<') ## previous character
		#########Insertion
		for h in extensions:
			editsR(h,tail,d-1,edit(pre+h[-1],pre))
		if not tail: return
		########Deletion
		editsR(head,tail[1:],d-1,edit(pre,pre+tail[0]))
		for h in extensions:
			if h[-1] == tail[0]: ##match
				editsR(h,tail[1:],d,edits)
			else: ##relacement
				editsR(h,tail[1:],d-1,edit(h[-1],tail[0]))
		##transpose

	editsR('',w,dis,[])
	return res



