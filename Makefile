# Convert org files to jupyter notebooks - requires pandoc
NOTEBOOKDIR := notebooks
VPATH = org:

NOTEBOOKS := $(addprefix $(NOTEBOOKDIR)/,MLP.ipynb XOR.ipynb)

$(NOTEBOOKDIR)/%.ipynb : %.org
	pandoc -f org -t ipynb -o $@ $<

.PHONY: all
all: $(NOTEBOOKS)

$(NOTEBOOKS): | $(NOTEBOOKDIR)

$(NOTEBOOKDIR):
	mkdir $(NOTEBOOKDIR)

.PHONY: clean
clean:
	rm $(NOTEBOOKS)
