import torch
import gradio as gr
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

def main():
    model = VisionEncoderDecoderModel.from_pretrained("OmkarShidore/scene-caption")
    feature_extractor = ViTImageProcessor.from_pretrained("OmkarShidore/scene-caption")
    tokenizer = AutoTokenizer.from_pretrained("OmkarShidore/scene-caption")
    
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    def predict(image):
        #image = Image.open(image_path)
        image = image.convert(mode="RGB")
        pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device="cpu")
        output_ids = model.generate(pixel_values, **gen_kwargs)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds[0]
    
    #built interface with gradio to test the function
    imagein = gr.components.Image(label='Scene Image', type='pil')
    output = gr.components.Textbox() 
    gui = gr.Interface(fn=predict, inputs=imagein, outputs=[output])

    gr.Interface(fn=predict,
             inputs=imagein,
             outputs=output,
             title='Image To Text- Scene Description',
             description="<html> <body> <h3>Hugging Face:  <a href='https://huggingface.co/OmkarShidore/scene-caption'>OmkarShidore/scene-caption</a></h3><h3>Git:  <a href='https://github.com/OmkarShidore/ImageToText-SceneDescription'>OmkarShidore/ImageToText-SceneDescription</a></h3> </body></html>",
             examples=["./data/car.jpg", "./data/gsd.jpg", "./data/highway.jpg"],
             theme=gr.themes.Base()
            ).launch(share=True);

if __name__ == '__main__':
    main()