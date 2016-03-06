#!/bin/sh
corpora=corpora/
url=http://faculty.cooper.edu/sable2/courses/spring2016/ece467/TC_provided.tar.gz

if [ -d $corpora ]; then
  >&2 echo Corpora already exist in $(pwd)/$corpora
  exit
fi

filename="$(basename $url)"
tmpdir="$(mktemp -d)"
wget -P $tmpdir $url
mkdir $corpora
tar -xzf $tmpdir/$filename -C $corpora --strip-components 1
rm -r $tmpdir
