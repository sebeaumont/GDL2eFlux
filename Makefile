# Convert org files to jupyter notebooks - requires pandoc
NOTEBOOKDIR := notebooks
VPATH = org:

TARGETS := MLP.ipynb XOR.ipynb

NOTEBOOKS := $(addprefix $(NOTEBOOKDIR)/,$(TARGETS))

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
