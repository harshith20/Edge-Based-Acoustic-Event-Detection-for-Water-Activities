import 'package:flutter/material.dart';
import 'package:record/record.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:path_provider/path_provider.dart';
import 'dart:io';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: RecorderPage(),
    );
  }
}

class RecorderPage extends StatefulWidget {
  const RecorderPage({super.key});

  @override
  State<RecorderPage> createState() => _RecorderPageState();
}

class _RecorderPageState extends State<RecorderPage> {
  final recorder = AudioRecorder();

  String status = "Idle";
  bool isRecording = false;
  String? filePath;

  // 🔥 START RECORDING (FIXED PATH)
  Future<void> startRecording() async {
    try {
      if (isRecording) return;

      if (await recorder.hasPermission()) {
        final dir = await getTemporaryDirectory();
        filePath = '${dir.path}/audio.wav';

        await recorder.start(
          const RecordConfig(
            encoder: AudioEncoder.wav,
          ),
          path: filePath!,
        );

        setState(() {
          status = "Recording...";
          isRecording = true;
        });
      } else {
        setState(() => status = "Microphone permission denied");
      }
    } catch (e) {
      setState(() => status = "Start error: $e");
    }
  }

  // 🔥 SEND AUDIO TO BACKEND
  Future<void> sendAudio(String path) async {
    try {
      var uri = Uri.parse("http://10.217.67.142:5001/predict");

      var request = http.MultipartRequest('POST', uri);
      request.files.add(await http.MultipartFile.fromPath('file', path));

      var response =
          await request.send().timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        var responseData = await response.stream.bytesToString();
        var jsonData = jsonDecode(responseData);

        setState(() {
          status =
              "Activity: ${jsonData['activity']}\nConfidence: ${jsonData['confidence']}";
        });
      } else {
        setState(() => status = "Server error: ${response.statusCode}");
      }
    } catch (e) {
      setState(() => status = "Network error: $e");
    }
  }

  // 🔥 STOP RECORDING + PROCESS (FIXED FILE CHECK)
  Future<void> stopRecording() async {
    try {
      if (!isRecording) return;

      final path = await recorder.stop();

      setState(() {
        status = "Processing...";
        isRecording = false;
      });

      if (path != null && File(path).existsSync()) {
        await sendAudio(path);
      } else {
        setState(() => status = "File not found");
      }
    } catch (e) {
      setState(() => status = "Stop error: $e");
    }
  }

  @override
  void dispose() {
    recorder.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Water Detection"),
        centerTitle: true,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                status,
                style: const TextStyle(fontSize: 18),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 30),

              ElevatedButton(
                onPressed: isRecording ? null : startRecording,
                child: const Text("Start Recording"),
              ),

              const SizedBox(height: 10),

              ElevatedButton(
                onPressed: isRecording ? stopRecording : null,
                child: const Text("Stop Recording"),
              ),
            ],
          ),
        ),
      ),
    );
  }
}