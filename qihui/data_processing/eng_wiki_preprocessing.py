 # TODO: preprocess the text of wiki, get data form as one line one sentence
"""
<A> ...</A> or <...>format info, delete
{{...}} citation info, delete (Warning: could be multi-lines)
[[...|...]] entitiies [[ redicted | running tokens ]]
== ... == Title / Subtitle, delete
&quot; quote \"
&amp; and &
&lt; ... &gt...&lt;/...&gt; non-compiled <..>, delete
"""
import re

directory_path = '/work/smt3/wwang/TAC2019/qihui_data/wiki/'

def preprocess_wiki(filename):

    with open(directory_path + filename, 'r') as fin:
        content = fin.read()
        content = re.sub(r"<.*?>.*?</.*?>", "", content)
        content = re.sub(r"&lt;.*?&gt.*?&lt;/.*?&gt;","", content)
        content = re.sub(r"<.*?>", "", content, flags=re.S)
        content = re.sub(r"\{{2}.*?\}{2}", "", content, flags=re.S)
        content = re.sub(r"\=\=.*?\=\=\n","\n", content)
        content = re.sub(r"&lt;.*?&gt;","", content, flags=re.S)
        content = re.sub("&quot;", "\"", content)
        content = re.sub("&amp;", "&", content)
    with open(directory_path + 'test_output_2', 'w') as fout:
        fout.write(content)

preprocess_wiki('test')


