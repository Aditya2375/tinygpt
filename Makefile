JAVAC = javac
JAVA = java -Xmx2g
CP = build/classes

all: classes

classes:
	mkdir -p $(CP)
	$(JAVAC) -d $(CP) src/gpt/token/Bpe.java src/gpt/model/Gpt.java src/gpt/train/Train.java tools/gpt/train/SampleInit.java tools/ExtractCorpus.java

test: classes
	$(JAVAC) -d $(CP) -cp $(CP) tests/gpt/tests/TestRunner.java
	$(JAVA) -cp $(CP) gpt.tests.TestRunner
	node tests/parity.mjs

train: classes
	$(JAVA) -cp $(CP) gpt.train.Train data/corpus.txt data/out 900 16

web:
	node build-web.mjs

clean:
	rm -rf build docs

.PHONY: all classes test train web clean
