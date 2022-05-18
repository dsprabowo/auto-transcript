from django.shortcuts import render
from django.conf import settings as django_settings
from django.core.files.storage import FileSystemStorage

from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
import speech_recognition as sr
import pandas as pd
import math
import numpy as np
import time
import re
import json

def index(request):
	context={'a':'Hello World!'}
	return render(request,'index.html',context)


def auto_transcript(request):

	start = time.perf_counter()
	# Upload audio from input
	media_url = django_settings.MEDIA_ROOT
	is_exist = os.path.exists(media_url)
	if not is_exist:
		os.makedirs(media_url)
	myfile = request.FILES['AudioInput']
	fs = FileSystemStorage()

	# save temporary file for processing
	filename = fs.save(myfile.name, myfile)
	
	
	file_path = os.path.join(media_url,str(filename))
	
	sampling_rate = 16000

	# load pyannote diarization pipeline
	pipeline_diarization = Pipeline.from_pretrained("pyannote/speaker-diarization")
	r = sr.Recognizer()

	# create audio segment from temporary audio, label speaker diarization with timestamps
	def diarization_segment(filepath,method):
		pipeline = pipeline_diarization   
		print('Start diarization')
		output = pipeline(filepath)
		json = output.for_json()
		print('Create labelling')
		segments = []
		for segment in json['content']:
			start = segment['segment']['start']
			end = segment['segment']['end']
			label = segment['label']
			temp = [start,end,label]
			segments.append(temp)
		print('Diarization finished')
		return(segments)
	
	# load audio file then recognize audio using google recognizer
	def audio_transcript(segments,filepath):
		transcript = []
		sound_file = AudioSegment.from_wav(filepath)
		print('Start transcription')
		for i,label in enumerate(segments):
			if i == 1:
				# buffer for 0.5 second
				test = sound_file[(label[0])*1000:(label[1]+0.5)*1000]
				filename = 'test_'+ str(i) +'.wav'
				savepath = os.path.join(media_url,filename)
				test.export(savepath,format='wav')
				audio_import = sr.AudioFile(savepath)
				with audio_import as source:
					audio = r.record(source)
				try:    
					text = r.recognize_google(audio, language = 'id-ID')
					output = [label[2],label[0],label[1],text]
					transcript.append(output)
				except:
					print('Segment '+ str(i) + ' inaudible')
					text = 'inaudible'
					output = [label[2],label[0],label[1],text]
					transcript.append(output)
				os.remove(savepath)
				print('Segment '+ str(i) + ' transcripted')
			else:
				# buffer for 0.5 second
				test = sound_file[(label[0]-0.5)*1000:(label[1]+0.5)*1000]
				filename = 'test_'+ str(i) +'.wav'
				savepath = os.path.join(media_url,filename)
				test.export(savepath,format='wav')
				audio_import = sr.AudioFile(savepath)
				with audio_import as source:
					audio = r.record(source)
				try:    
					text = r.recognize_google(audio, language = 'id-ID')
					output = [label[2],label[0],label[1],text]
					transcript.append(output)
				except:
					print('Segment '+ str(i) + ' inaudible')
					text = 'inaudible'
					output = [label[2],label[0],label[1],text]
					transcript.append(output)
				os.remove(savepath)
				print('Segment '+ str(i) + ' transcripted')
		print('Transcription finished')
		return(transcript)

	# wrapper function for diarization_segment and audio_transcript
	def auto_transcript(filepath, method):
		segment =  diarization_segment(filepath,method)
		full_transcript = audio_transcript(segment,filepath)
		return(full_transcript)

	# run whole pipeline with temporary audio file
	temp = auto_transcript(file_path, 'diarization')

	# remove temporary audio file
	os.remove(file_path)
	
	# convert output into dataframe format
	temp = pd.DataFrame(temp, columns = ['speaker', 'start','end','text'])

	# round timestamps into two decimals
	temp['start'] = round(temp['start'],2)
	temp['end'] = round(temp['end'],2)

	# prepare html output
	temp = temp.to_html()
	context = {'result': temp}
	
	end = time.perf_counter()
	print('Start time:', start)
	print('End time:', end)
	print('Time elapsed :', end - start)
	return render(request, 'index.html', context)