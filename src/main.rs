use async_openai::types::AudioInput;
use async_openai::{types::CreateTranscriptionRequestArgs, Client};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct InputDevice {
    pub host: cpal::Host,
    pub device: cpal::Device,
    pub config: cpal::SupportedStreamConfig,
}

// The WAV file we're recording to.
const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/recorded.wav");

impl InputDevice {
    pub fn new() -> anyhow::Result<Self> {
        let host = cpal::default_host();

        let device = host
            .default_input_device()
            // TODO: replace unwrap later
            .ok_or("No input device available")
            .unwrap();

        let config = device.default_input_config()?;

        Ok(Self {
            host,
            device,
            config,
        })
    }
    pub fn get_all(&self) -> (&cpal::Host, &cpal::Device, &cpal::SupportedStreamConfig) {
        (&self.host, &self.device, &self.config)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let input_device = InputDevice::new()?;

    let spec = wav_spec_from_config(&input_device.config);
    let writer = hound::WavWriter::create(PATH, spec)?;
    let writer = Arc::new(Mutex::new(Some(writer)));
    // A flag to indicate that recording is in progress.
    println!("Begin recording...");

    // Run the input stream on a separate thread.
    let writer_2 = writer.clone();

    write_mic_audio_to_file(writer_2, &input_device, Duration::from_secs(5))?;

    let filename = PATH.to_string();
    let file_contents = std::fs::read(filename)?;

    let audio_input = AudioInput::from_vec_u8(PATH.to_string(), file_contents);
    let client = Client::new();
    let request = CreateTranscriptionRequestArgs::default()
        .file(audio_input)
        .model("whisper-1")
        .build()?;
    let response = client.audio().transcribe(request).await?;
    println!("{}", response.text);
    std::fs::remove_file(PATH)?;
    Ok(())
}

fn write_mic_audio_to_file(
    wav_file_writer: Arc<Mutex<Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>>>>,
    input_device: &InputDevice,
    duration: Duration,
) -> Result<(), anyhow::Error> {
    let (_, device, config) = input_device.get_all();

    let sample_format = config.sample_format();

    println!("sample format {sample_format:?}");

    // let stream_data = Arc::new(Mutex::new(Vec::new()));
    // let stream_data_clone = Arc::clone(&stream_data);

    let err_fn = |err| eprintln!("An error occurred on the audio stream: {}", err);

    let wav_file_writer = wav_file_writer.clone();
    let wav_file_writer_clone = wav_file_writer.clone();

    let stream = match config.sample_format() {
        cpal::SampleFormat::I8 => device.build_input_stream(
            &config.to_owned().into(),
            move |data, _: &_| write_input_data::<i8, i8>(data, &wav_file_writer),
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config.to_owned().into(),
            move |data, _: &_| write_input_data::<i16, i16>(data, &wav_file_writer),
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I32 => device.build_input_stream(
            &config.to_owned().into(),
            move |data, _: &_| write_input_data::<i32, i32>(data, &wav_file_writer),
            err_fn,
            None,
        )?,
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.to_owned().into(),
            move |data, _: &_| write_input_data::<f32, f32>(data, &wav_file_writer),
            err_fn,
            None,
        )?,
        sample_format => {
            return Err(anyhow::Error::msg(format!(
                "Unsupported sample format '{sample_format}'"
            )))
        }
    };

    stream.play()?;

    // Let recording go for roughly three seconds.
    std::thread::sleep(duration);
    drop(stream);
    wav_file_writer_clone
        .lock()
        .unwrap()
        .take()
        .unwrap()
        .finalize()?;
    println!("Recording {} complete!", PATH);
    Ok(())
}

fn wav_spec_from_config(config: &cpal::SupportedStreamConfig) -> hound::WavSpec {
    hound::WavSpec {
        channels: config.channels() as _,
        sample_rate: config.sample_rate().0 as _,
        bits_per_sample: (config.sample_format().sample_size() * 8) as _,
        sample_format: sample_format(config.sample_format()),
    }
}

fn sample_format(format: cpal::SampleFormat) -> hound::SampleFormat {
    if format.is_float() {
        hound::SampleFormat::Float
    } else {
        hound::SampleFormat::Int
    }
}

type WavWriterHandle = Arc<Mutex<Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>>>>;

fn write_input_data<T, U>(input: &[T], writer: &WavWriterHandle)
where
    T: cpal::Sample,
    U: cpal::Sample + hound::Sample + cpal::FromSample<T>,
{
    if let Ok(mut guard) = writer.try_lock() {
        if let Some(writer) = guard.as_mut() {
            for &sample in input.iter() {
                let sample: U = U::from_sample(sample);
                writer.write_sample(sample).ok();
            }
        }
    }
}
