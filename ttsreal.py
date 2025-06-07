###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
from __future__ import annotations
import time
import numpy as np
import soundfile as sf
import resampy
import asyncio
import edge_tts

import os
import hmac
import hashlib
import base64
import json
import uuid

from typing import Iterator

import requests

import queue
from queue import Queue
from io import BytesIO
from threading import Thread, Event
from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from basereal import BaseReal

from logger import logger
class State(Enum):
    RUNNING=0
    PAUSE=1

class BaseTTS:
    def __init__(self, opt, parent:BaseReal):
        self.opt=opt
        self.parent = parent

        self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        self.msgqueue = Queue()
        self.state = State.RUNNING

    def flush_talk(self):
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self,msg:str,eventpoint=None): 
        if len(msg)>0:
            self.msgqueue.put((msg,eventpoint))

    def render(self,quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()
    
    def process_tts(self,quit_event):        
        while not quit_event.is_set():
            try:
                msg = self.msgqueue.get(block=True, timeout=1)
                self.state=State.RUNNING
            except queue.Empty:
                continue
            self.txt_to_audio(msg)
        logger.info('ttsreal thread stop')
    
    def txt_to_audio(self,msg):
        pass
    

###########################################################################################
class EdgeTTS(BaseTTS):
    def txt_to_audio(self,msg):
        voicename = self.opt.REF_FILE #"zh-CN-YunxiaNeural"
        text,textevent = msg
        t = time.time()
        asyncio.new_event_loop().run_until_complete(self.__main(voicename,text))
        logger.info(f'-------edge tts time:{time.time()-t:.4f}s')
        if self.input_stream.getbuffer().nbytes<=0: #edgetts err
            logger.error('edgetts err!!!!!')
            return
        
        self.input_stream.seek(0)
        stream = self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk and self.state==State.RUNNING:
            eventpoint=None
            streamlen -= self.chunk
            if idx==0:
                eventpoint={'status':'start','text':text,'msgevent':textevent}
            elif streamlen<self.chunk:
                eventpoint={'status':'end','text':text,'msgevent':textevent}
            self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
            idx += self.chunk
        #if streamlen>0:  #skip last frame(not 20ms)
        #    self.queue.put(stream[idx:])
        self.input_stream.seek(0)
        self.input_stream.truncate() 

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream
    
    async def __main(self,voicename: str, text: str):
        try:
            communicate = edge_tts.Communicate(text, voicename)

            #with open(OUTPUT_FILE, "wb") as file:
            first = True
            async for chunk in communicate.stream():
                if first:
                    first = False
                if chunk["type"] == "audio" and self.state==State.RUNNING:
                    #self.push_audio(chunk["data"])
                    self.input_stream.write(chunk["data"])
                    #file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
        except Exception as e:
            logger.exception('edgetts')

###########################################################################################
class FishTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.api_key = "ffb55690a2c74be39af987075cde4a86"  # fish.audio API key
        self.model_id = "faccba1a8ac54016bcfc02761285e67f"  # fish.audio model ID
        logger.info(f"[FISH_TTS] 初始化完成 - API Key: {self.api_key[:10]}..., Model ID: {self.model_id}")
        
    def txt_to_audio(self,msg): 
        text,textevent = msg
        logger.info(f"[FISH_TTS] 开始处理文本: '{text}', 事件: {textevent}")
        self.stream_tts(
            self.fish_audio_api(
                text,
                self.model_id,
                self.api_key
            ),
            msg
        )

    def fish_audio_api(self, text, model_id, api_key) -> Iterator[bytes]:
        logger.info(f"[FISH_TTS] 调用Fish.Audio API - 文本: '{text}', 模型: {model_id}")
        start = time.perf_counter()
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        
        # 使用MP3格式，因为WAV格式返回空数据
        payload = {
            'text': text,
            'reference_id': model_id,
            # 不指定format，默认返回MP3（有音频数据）
            # 'format': 'wav',  # WAV格式返回空数据
        }
        
        logger.info(f"[FISH_TTS] 发送请求参数: {payload}")
        
        try:
            res = requests.post(
                "https://api.fish.audio/v1/tts",
                json=payload,
                stream=False,  # 先不使用流式传输
                headers=headers,
                timeout=30
            )
            end = time.perf_counter()
            logger.info(f"[FISH_TTS] API响应 - 状态码: {res.status_code}, 耗时: {end-start:.2f}s")

            if res.status_code != 200:
                logger.error("[FISH_TTS] API错误 - 状态码: %d, 响应: %s", res.status_code, res.text)
                return
                
            content_length = len(res.content)
            logger.info(f"[FISH_TTS] API成功 - 音频数据大小: {content_length}字节")
            
            # 检查返回的内容类型
            content_type = res.headers.get('content-type', '')
            logger.info(f"[FISH_TTS] 响应内容类型: {content_type}")
            
            # 直接返回完整的音频数据
            if res.content and len(res.content) > 0:
                yield res.content
            else:
                logger.warning("[FISH_TTS] API返回空内容")
                    
        except Exception as e:
            logger.exception('[FISH_TTS] API调用异常')

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        logger.info(f"[FISH_TTS] 开始音频流处理 - 文本: '{text}'")
        first = True
        audio_buffer = BytesIO()
        
        # 收集所有音频数据
        chunk_count = 0
        for chunk in audio_stream:
            if chunk is not None and len(chunk) > 0:
                audio_buffer.write(chunk)
                chunk_count += 1
                logger.info(f"[FISH_TTS] 收到音频块 {chunk_count}, 大小: {len(chunk)}字节")
        
        logger.info(f"[FISH_TTS] 总共收到 {chunk_count} 个音频块, 总大小: {audio_buffer.getbuffer().nbytes}字节")
        
        # 处理完整的音频数据
        if audio_buffer.getbuffer().nbytes > 0:
            # 先保存原始音频数据用于调试
            debug_filename = f"debug_audio_{int(time.time())}.mp3"  # 改为.mp3
            with open(debug_filename, 'wb') as f:
                f.write(audio_buffer.getvalue())
            logger.info(f"[FISH_TTS] 已保存调试音频文件: {debug_filename}")
            
            stream = None
            sample_rate = None
            
            # 优先使用pydub处理MP3格式
            try:
                from pydub import AudioSegment
                import numpy as np
                
                # 检测音频格式
                audio_buffer.seek(0)
                header = audio_buffer.read(4)
                audio_buffer.seek(0)
                
                if header.startswith(b'\xff\xfb') or header.startswith(b'ID3'):
                    # MP3格式
                    logger.info(f"[FISH_TTS] 检测到MP3格式")
                    audio_segment = AudioSegment.from_mp3(audio_buffer)
                elif header.startswith(b'RIFF'):
                    # WAV格式
                    logger.info(f"[FISH_TTS] 检测到WAV格式")
                    audio_segment = AudioSegment.from_wav(audio_buffer)
                else:
                    # 尝试自动检测
                    logger.info(f"[FISH_TTS] 尝试自动检测格式: {header.hex()}")
                    audio_buffer.seek(0)
                    audio_segment = AudioSegment.from_file(audio_buffer)
                
                if len(audio_segment) > 0:  # 检查音频时长
                    # 转换为numpy数组
                    samples = audio_segment.get_array_of_samples()
                    stream = np.array(samples, dtype=np.float32)
                    
                    # 归一化
                    if audio_segment.sample_width == 1:  # 8-bit
                        stream = (stream - 128) / 127.0
                    elif audio_segment.sample_width == 2:  # 16-bit
                        stream = stream / 32767.0
                    elif audio_segment.sample_width == 4:  # 32-bit
                        stream = stream / 2147483647.0
                    
                    sample_rate = audio_segment.frame_rate
                    
                    # 处理立体声
                    if audio_segment.channels == 2:
                        stream = stream.reshape((-1, 2))
                        
                    logger.info(f'[FISH_TTS] pydub解析成功 - 采样率: {sample_rate}, 形状: {stream.shape}, 声道: {audio_segment.channels}, 时长: {len(audio_segment)}ms')
                    
                else:
                    logger.warning(f'[FISH_TTS] pydub返回空音频 - 时长: {len(audio_segment)}ms')
                    raise ValueError("pydub返回空音频数据")
                        
            except Exception as e_pydub:
                logger.warning(f'[FISH_TTS] pydub解析失败: {e_pydub}')
                
                # 备用方案：尝试librosa
                try:
                    import librosa
                    audio_buffer.seek(0)
                    stream_temp, sample_rate_temp = librosa.load(audio_buffer, sr=None)
                    if stream_temp.shape[0] > 0:
                        stream = stream_temp
                        sample_rate = sample_rate_temp
                        logger.info(f'[FISH_TTS] librosa解析成功 - 采样率: {sample_rate}, 形状: {stream.shape}')
                    else:
                        logger.warning(f'[FISH_TTS] librosa返回空数据 - 采样率: {sample_rate_temp}, 形状: {stream_temp.shape}')
                        raise ValueError("librosa返回空音频数据")
                except Exception as e_librosa:
                    logger.error(f'[FISH_TTS] 所有音频解析方法都失败: pydub={e_pydub}, librosa={e_librosa}')
                    
                    # 保持调试文件以供手动检查
                    logger.info(f'[FISH_TTS] 保留调试文件 {debug_filename} 以供手动检查')
                    return
            
            # 验证最终音频数据有效性
            if stream is None or stream.shape[0] == 0:
                logger.error(f'[FISH_TTS] 最终音频数据无效 - stream: {stream}, shape: {stream.shape if stream is not None else "None"}')
                return
            
            try:
                # 转换为float32
                stream = stream.astype(np.float32)
                
                # 处理多声道
                if stream.ndim > 1:
                    logger.info(f'[FISH_TTS] 多声道音频，使用第一声道 - 声道数: {stream.shape[1]}')
                    stream = stream[:, 0]
                
                # 重采样到目标采样率
                if sample_rate != self.sample_rate and stream.shape[0] > 0:
                    logger.info(f'[FISH_TTS] 重采样音频 - 从 {sample_rate}Hz 到 {self.sample_rate}Hz')
                    stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)
                
                # 分块发送音频
                streamlen = stream.shape[0]
                idx = 0
                frame_count = 0
                logger.info(f'[FISH_TTS] 开始发送音频帧 - 总长度: {streamlen}样本, 块大小: {self.chunk}')
                
                while streamlen >= self.chunk and self.state == State.RUNNING:
                    eventpoint = None
                    streamlen -= self.chunk
                    if idx == 0:
                        eventpoint = {'status': 'start', 'text': text, 'msgevent': textevent}
                        logger.info(f'[FISH_TTS] 发送开始事件')
                    elif streamlen < self.chunk:
                        eventpoint = {'status': 'end', 'text': text, 'msgevent': textevent}
                        logger.info(f'[FISH_TTS] 发送结束事件')
                    
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
                    idx += self.chunk
                    frame_count += 1
                    
                    if frame_count % 50 == 0:  # 每50帧记录一次
                        logger.info(f'[FISH_TTS] 已发送 {frame_count} 帧音频')
                
                logger.info(f'[FISH_TTS] 音频处理完成 - 总共发送 {frame_count} 帧')
                
                # 只有在成功处理后才清理调试文件
                try:
                    import os
                    if os.path.exists(debug_filename):
                        os.remove(debug_filename)
                        logger.info(f'[FISH_TTS] 已清理调试文件: {debug_filename}')
                except:
                    pass
                    
            except Exception as e:
                logger.exception('[FISH_TTS] 音频流处理异常')
        
        # 如果没有音频数据，发送结束事件
        if audio_buffer.getbuffer().nbytes == 0:
            logger.warning("[FISH_TTS] 没有收到音频数据，发送空结束事件")
            eventpoint = {'status': 'end', 'text': text, 'msgevent': textevent}
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

###########################################################################################
class SovitsTTS(BaseTTS):
    def txt_to_audio(self,msg): 
        text,textevent = msg
        self.stream_tts(
            self.gpt_sovits(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def gpt_sovits(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        req={
            'text':text,
            'text_lang':language,
            'ref_audio_path':reffile,
            'prompt_text':reftext,
            'prompt_lang':language,
            'media_type':'ogg',
            'streaming_mode':True
        }
        # req["text"] = text
        # req["text_language"] = language
        # req["character"] = character
        # req["emotion"] = emotion
        # #req["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        # req["streaming_mode"] = True
        try:
            res = requests.post(
                f"{server_url}/tts",
                json=req,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"gpt_sovits Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=None): #12800 1280 32K*20ms*2
                logger.info('chunk len:%d',len(chunk))
                if first:
                    end = time.perf_counter()
                    logger.info(f"gpt_sovits Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
            #print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            logger.exception('sovits')

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                #stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                #stream = resampy.resample(x=stream, sr_orig=32000, sr_new=self.sample_rate)
                byte_stream=BytesIO(chunk)
                stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgevent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgevent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)

###########################################################################################
class CosyVoiceTTS(BaseTTS):
    def txt_to_audio(self,msg):
        text,textevent = msg 
        self.stream_tts(
            self.cosy_voice(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def cosy_voice(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        payload = {
            'tts_text': text,
            'prompt_text': reftext
        }
        try:
            files = [('prompt_wav', ('prompt_wav', open(reffile, 'rb'), 'application/octet-stream'))]
            res = requests.request("GET", f"{server_url}/inference_zero_shot", data=payload, files=files, stream=True)
            
            end = time.perf_counter()
            logger.info(f"cosy_voice Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=9600): # 960 24K*20ms*2
                if first:
                    end = time.perf_counter()
                    logger.info(f"cosy_voice Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('cosyvoice')

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgevent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgevent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint) 

###########################################################################################
_PROTOCOL = "https://"
_HOST = "tts.cloud.tencent.com"
_PATH = "/stream"
_ACTION = "TextToStreamAudio"

class TencentTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt,parent)
        self.appid = os.getenv("TENCENT_APPID")
        self.secret_key = os.getenv("TENCENT_SECRET_KEY")
        self.secret_id = os.getenv("TENCENT_SECRET_ID")
        self.voice_type = int(opt.REF_FILE)
        self.codec = "pcm"
        self.sample_rate = 16000
        self.volume = 0
        self.speed = 0
    
    def __gen_signature(self, params):
        sort_dict = sorted(params.keys())
        sign_str = "POST" + _HOST + _PATH + "?"
        for key in sort_dict:
            sign_str = sign_str + key + "=" + str(params[key]) + '&'
        sign_str = sign_str[:-1]
        hmacstr = hmac.new(self.secret_key.encode('utf-8'),
                           sign_str.encode('utf-8'), hashlib.sha1).digest()
        s = base64.b64encode(hmacstr)
        s = s.decode('utf-8')
        return s

    def __gen_params(self, session_id, text):
        params = dict()
        params['Action'] = _ACTION
        params['AppId'] = int(self.appid)
        params['SecretId'] = self.secret_id
        params['ModelType'] = 1
        params['VoiceType'] = self.voice_type
        params['Codec'] = self.codec
        params['SampleRate'] = self.sample_rate
        params['Speed'] = self.speed
        params['Volume'] = self.volume
        params['SessionId'] = session_id
        params['Text'] = text

        timestamp = int(time.time())
        params['Timestamp'] = timestamp
        params['Expired'] = timestamp + 24 * 60 * 60
        return params

    def txt_to_audio(self,msg):
        text,textevent = msg 
        self.stream_tts(
            self.tencent_voice(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def tencent_voice(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        session_id = str(uuid.uuid1())
        params = self.__gen_params(session_id, text)
        signature = self.__gen_signature(params)
        headers = {
            "Content-Type": "application/json",
            "Authorization": str(signature)
        }
        url = _PROTOCOL + _HOST + _PATH
        try:
            res = requests.post(url, headers=headers,
                          data=json.dumps(params), stream=True)
            
            end = time.perf_counter()
            logger.info(f"tencent Time to make POST: {end-start}s")
                
            first = True
        
            for chunk in res.iter_content(chunk_size=6400): # 640 16K*20ms*2
                #logger.info('chunk len:%d',len(chunk))
                if first:
                    try:
                        rsp = json.loads(chunk)
                        #response["Code"] = rsp["Response"]["Error"]["Code"]
                        #response["Message"] = rsp["Response"]["Error"]["Message"]
                        logger.error("tencent tts:%s",rsp["Response"]["Error"]["Message"])
                        return
                    except:
                        end = time.perf_counter()
                        logger.info(f"tencent Time to first chunk: {end-start}s")
                        first = False                    
                if chunk and self.state==State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('tencent')

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        last_stream = np.array([],dtype=np.float32)
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = np.concatenate((last_stream,stream))
                #stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgevent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
                last_stream = stream[idx:] #get the remain stream
        eventpoint={'status':'end','text':text,'msgevent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint) 

###########################################################################################

class XTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt,parent)
        self.speaker = self.get_speaker(opt.REF_FILE, opt.TTS_SERVER)

    def txt_to_audio(self,msg):
        text,textevent = msg  
        self.stream_tts(
            self.xtts(
                text,
                self.speaker,
                "zh-cn", #en args.language,
                self.opt.TTS_SERVER, #"http://localhost:9000", #args.server_url,
                "20" #args.stream_chunk_size
            ),
            msg
        )

    def get_speaker(self,ref_audio,server_url):
        files = {"wav_file": ("reference.wav", open(ref_audio, "rb"))}
        response = requests.post(f"{server_url}/clone_speaker", files=files)
        return response.json()

    def xtts(self,text, speaker, language, server_url, stream_chunk_size) -> Iterator[bytes]:
        start = time.perf_counter()
        speaker["text"] = text
        speaker["language"] = language
        speaker["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        try:
            res = requests.post(
                f"{server_url}/tts_stream",
                json=speaker,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"xtts Time to make POST: {end-start}s")

            if res.status_code != 200:
                print("Error:", res.text)
                return

            first = True
        
            for chunk in res.iter_content(chunk_size=9600): #24K*20ms*2
                if first:
                    end = time.perf_counter()
                    logger.info(f"xtts Time to first chunk: {end-start}s")
                    first = False
                if chunk:
                    yield chunk
        except Exception as e:
            print(e)
    
    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgevent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgevent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)  