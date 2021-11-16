import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException

def triton_infer(batch_img):
    ## Client-Server
    ## server_url = ‘10.233.28.188:8000’
    server_url = 'triton-routes8000.hongyang.toolmenlab.bime.ntu.edu.tw'
    triton_client = httpclient.InferenceServerClient(url=server_url, verbose=False)
    ## Inference image
    ## Input setting
    # img = preprocessing('data/fish.jpg')
    inputs = []
    inputs.append(httpclient.InferInput('input_image', batch_img.shape, 'FP32'))
    inputs[0].set_data_from_numpy(batch_img)

    ## Output setting
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('output_conf'))
    outputs.append(httpclient.InferRequestedOutput('output_mask'))
    outputs.append(httpclient.InferRequestedOutput('output_loc'))
    outputs.append(httpclient.InferRequestedOutput('output_priors'))
    outputs.append(httpclient.InferRequestedOutput('output_proto'))


    ## Send request (Inference)
    result = triton_client.infer("yolact",
                                 inputs,
                                 model_version='1',
                                 request_id=str(1),
                                 outputs=outputs)
                                 
    results = {'conf': result.as_numpy('output_conf'),
               'mask': result.as_numpy('output_mask'),
               'loc': result.as_numpy('output_loc'),
               'priors': result.as_numpy('output_priors'),
               'proto': result.as_numpy('output_proto')}

    return results

if __name__ == '__main__':
    triton_infer(None)