// Live backend only: progressive flow, bounded computation, values and recorded playback.
import {chromium,expect} from '@playwright/test';
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
const page=await browser.newPage({viewport:{width:1700,height:1200}});const errors=[];
page.on('pageerror',e=>errors.push(String(e)));
const model=process.env.P2I_DEMO??'transformer';
try{
 await page.goto(process.env.P2I_URL??'http://127.0.0.1:8000');
 await expect(page.locator('.architecture-flow')).toBeVisible();
 const container=model==='transformer'?'blocks.0':model==='cnn'?'features':'layers';
 if(model==='transformer')await page.getByRole('button',{name:'Explore blocks',exact:true}).click();
 await page.getByRole('button',{name:`Explore ${container}`,exact:true}).click();
 await expect(page.locator('.module-flow-node').first()).toBeVisible();
 const ir=await (await page.request.get(new URL('/api/model',page.url()).href)).json();
 // Every visible module link must reference a tensor in the actual trace.
 const edgeLabels=await page.locator('.module-flow-edge').evaluateAll(es=>es.map(e=>e.getAttribute('aria-label')));
 for(const label of edgeLabels)if(!ir.tensors.some(t=>`Inspect tensor ${t.id}`===label))throw Error('Unobserved tensor route');
 await page.screenshot({path:`../docs/ux-${model}-explore.png`,fullPage:true});
 await page.getByRole('button',{name:'Computation',exact:true}).click();
 await expect(page.locator('.operation-explainer')).toBeVisible();
 if(await page.locator('.op-node').count()>7)throw Error('Default computation is not bounded');
 await expect(page.locator('.graph-controls')).toContainText('100%');
 const before=await page.getByLabel('Focused operation').inputValue();
 if(await page.getByRole('button',{name:'Next operation',exact:true}).isEnabled()){
  await page.getByRole('button',{name:'Next operation',exact:true}).click();
  if(await page.getByLabel('Focused operation').inputValue()===before)throw Error('Operation navigation failed');
 }
 await page.locator('.operation-tensors button').first().click();
 await expect(page.locator('.details')).toContainText('Shape');
 await page.getByRole('button',{name:'Fit scope',exact:true}).click();
 const percent=await page.locator('.graph-controls').textContent();
 if(Number(percent.match(/(\d+)%/)[1])<80)throw Error('Fit shrank readable nodes');
 await page.getByRole('button',{name:'Raw graph',exact:true}).click();
 await expect(page.getByRole('button',{name:'Focused view',exact:true})).toBeVisible();
 await page.getByRole('button',{name:'Focused view',exact:true}).click();
 // Opt-in values on the real model. No sample values are mocked.
 await page.getByRole('button',{name:'Edit',exact:true}).click();
 await page.getByLabel('Capture bounded CPU tensor samples on re-trace (opt-in)').check();
 await page.getByRole('button',{name:'Build & Re-trace',exact:true}).click();
 await expect(page.locator('.validation-result')).toContainText('Validation passed',{timeout:30000});
 await page.getByRole('button',{name:'Computation',exact:true}).click();
 await expect(page.locator('.heatmap').first()).toBeVisible();
 await page.locator('.heatmap button').first().hover();
 await expect(page.locator('.sample-readout').first()).toContainText('flat[0]');
 const samples=await (await page.request.get(new URL('/api/inspection',page.url()).href)).json();
 const sampleButtons=await page.locator('.heatmap').first().getByRole('button').all();
 if(sampleButtons.length>64)throw Error('Tensor sample rendering exceeds cap');
 if(!Object.values(samples.tensors).some(s=>s.values?.length))throw Error('No real samples captured');
 await page.screenshot({path:`../docs/ux-${model}-compute.png`,fullPage:true});
 await page.getByRole('button',{name:'Runtime',exact:true}).click();
 await expect(page.locator('.execution-active')).toHaveCount(1);
 if(await page.getByRole('button',{name:'Next',exact:true}).isEnabled())await page.getByRole('button',{name:'Next',exact:true}).click();
 await expect(page.locator('.execution-active')).toHaveCount(1);
 const active=await page.locator('.current-call h2').textContent();
 await expect(page.locator('.execution-active')).toContainText(active.split('.').at(-1));
 const callBeforePlay=await page.locator('.current-call>span').textContent();
 await page.getByRole('button',{name:'Play',exact:true}).click();
 await expect(page.getByRole('button',{name:'Pause',exact:true})).toBeVisible();
 await expect(page.locator('.current-call>span')).not.toHaveText(callBeforePlay);
 await page.getByRole('button',{name:'Pause',exact:true}).click();
 await page.screenshot({path:`../docs/ux-${model}-runtime.png`,fullPage:true});
 if(errors.length)throw Error(errors.join('\n'));
 console.log(`PASS UX ${model}: observed module routes, <=7 node neighborhood, readable fit, raw mode, operation explanations, real tensor samples, synchronized recorded playback`);
}finally{await browser.close()}
