import json
import sys
from pathlib import Path
from src.config import load_settings
from src.rag import RagPipeline

p = RagPipeline(load_settings(knowledge_base='anilist'))
cases = [
 ('Recommend me a Short Anime', [('episodes','lte','50')], True),
 ('Suggest a brief series', [('episodes','lte','50')], True),
 ('Give me anime that was released at the 90s', [('year','gte','1990'),('year','lte','1999')], False),
 ('Recommend anime released in the eighties', [('year','gte','1980'),('year','lte','1989')], False),
 ('Suggest anime from the 2000s', [('year','gte','2000'),('year','lte','2009')], False),
 ('Recommend short fantasy anime released in the 1990s', [('episodes','lte','50'),('year','gte','1990'),('year','lte','1999'),('genres','eq','Fantasy')], True),
 ('Recommend fantasy TV anime with at most 12 episodes released since 2015, preferably funny',
  [('episodes','lte','12'),('year','gte','2015'),('format','eq','TV'),('genres','eq','Fantasy')], False),
 ('Recommend anime with episodes under 20 minutes', [('duration','lt','20')], False),
 ("I'd like an Action Anime that doesn't have romance", [('genres','eq','Action'),('genres','ne','Romance')], False),
 ("Recommend me some romance anime that wasn't set in high school and are emotional", [('genres','eq','Romance'),('tags','ne','School')], False),
 ('Recommend anime released before the 1980s', [('year','lt','1980')], False),
 ('Recommend movies with runtime under 2 hours', [('format','eq','MOVIE'),('duration','lt','120')], False),
]
path = Path('data/semantic-request-regression.json')
selected = set(map(int,sys.argv[1:])) if len(sys.argv)>1 else set(range(len(cases)))
results = json.loads(path.read_text(encoding='utf-8')) if path.exists() and len(sys.argv)>1 else []
for index,(question,expected,short) in enumerate(cases):
 if index not in selected: continue
 entry={'question':question}
 print('RUNNING '+question,flush=True)
 try:
  entry['answer']=p.ask(question)
  request=p.last_route['request_constraints']
  actual={(f['field'],f['operator'],f['value']) for f in request['filters']}
  assert actual==set(expected), f'Wrong filters: {actual}'
  items={item['id']:p.anime_tools().record(item['id']) for trace in p.last_route['agent_trace']
         for item in trace.get('result',{}).get('items',[]) if trace.get('tool') in ('recommend_anime','popular')}
  assert items,'No results'
  entry['records']=[{key:r.get(key) for key in ('id','title','episodes','duration','year')} for r in items.values()]
  where,params=p.anime_tools()._where({'filters':request['filters']})
  for aid in items:
   assert p.anime_tools().query('SELECT a.id FROM anime a WHERE a.id=? AND '+where,[aid,*params])
  if short:
   assert all(0<r['episodes']<=50 for r in items.values()),'Episode ceiling violated'
   counts=[10<=r['episodes']<=19 for r in items.values()]
   assert any(counts),'No preferred episode counts'
   assert counts==sorted(counts,reverse=True),'Preferred counts did not rank first'
  entry['validation']='PASS'
 except Exception as error:
  entry['validation']='FAIL';entry['error']=str(error)
 entry['routing']=p.last_route
 if path.exists():
  results=json.loads(path.read_text(encoding='utf-8'))
 results=[row for row in results if row['question']!=question]+[entry]
 path.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({k:entry[k] for k in ('question','validation','error') if k in entry}),flush=True)
