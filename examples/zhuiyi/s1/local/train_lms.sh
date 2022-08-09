#!/bin/bash
# Created by yuanding on 2022/07/26
# 构建语言模型.

set -e
. ./path.sh

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <text> <lexicon> <lm_dir>"
  echo "text: 文本路径."
  echo "lexicon: 词典路径."
  echo "lm_dir: 语言模型文件夹, 用来保存arpa模型和相关文件."
  exit 1
fi

text=$1
lexicon=$2
dir=$3

for f in "$text" "$lexicon"; do
  [ ! -f $x ] && echo "$0: No such file $f" && exit 1;
done

# Check SRILM tools
if ! which ngram-count > /dev/null; then
    echo "srilm tools are not found, please download it and install it from: "
    echo "http://www.speech.sri.com/projects/srilm/download.html"
    echo "Then add the tools to your PATH"
    exit 1
fi

mkdir -p $dir

cleantext=$dir/text.no_oov

cat $text | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } }
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<SPOKEN_NOISE> ");} } printf("\n");}' \
  > $cleantext || exit 1;

cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | sort | uniq -c | \
   sort -nr > $dir/word.counts || exit 1;

cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | \
  cat - <(grep -w -v '!SIL' $lexicon | awk '{print $1}') | \
   sort | uniq -c | sort -nr > $dir/unigram.counts || exit 1;

cat $dir/unigram.counts | awk '{print $2}' | cat - <(echo "<s>"; echo "</s>" ) > $dir/wordlist

cp $cleantext  $dir/train

ngram-count -text $dir/train -order 3 -limit-vocab -vocab $dir/wordlist -unk \
  -map-unk "<UNK>" -kndiscount -interpolate -lm $dir/lm.arpa
