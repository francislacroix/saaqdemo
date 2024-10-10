using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Primitives;

using Azure;
using Azure.AI.Vision.Face;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

namespace My.Company
{
    public class ProcessImages
    {
        static readonly string APIKEY = Environment.GetEnvironmentVariable("FACE_APIKEY") ?? "<apikey>";
        static readonly string ENDPOINT = Environment.GetEnvironmentVariable("FACE_ENDPOINT") ?? "<endpoint>";

        private readonly ILogger<ProcessImages> _logger;

        public ProcessImages(ILogger<ProcessImages> logger)
        {
            _logger = logger;
        }

        [Function("TraiterImages")]
        public async Task<IActionResult> Run([HttpTrigger(AuthorizationLevel.Function, "get", "post")] HttpRequest req)
        {
            FaceClient faceClient = new FaceClient(new Uri(ENDPOINT), new AzureKeyCredential(APIKEY));

            if (req.Query.TryGetValue("imageUrl", out StringValues imageUrl))
            {
                Response<IReadOnlyList<FaceDetectionResult>> faceDetectResponse = await faceClient.DetectAsync(new Uri(imageUrl), FaceDetectionModel.Detection03, FaceRecognitionModel.Recognition04, false, [FaceAttributeType.QualityForRecognition]);
                IReadOnlyList<FaceDetectionResult> detectedFaces = faceDetectResponse.Value;

                if (detectedFaces.Count == 0)
                {
                    return new BadRequestObjectResult("No faces detected in the image");
                }
                else if (detectedFaces.Count > 1)
                {
                    return new BadRequestObjectResult("Multiple faces detected in the image");
                }
                else
                {
                    FaceDetectionResult detectedFace = detectedFaces[0];
                    var faceQualityForRecognition = detectedFace.FaceAttributes.QualityForRecognition;

                    if (faceQualityForRecognition.HasValue && (faceQualityForRecognition.Value != QualityForRecognition.Low))
                    {
                        HttpClient httpClient = new HttpClient();

                        // Download the image as a stream 
                        Stream imageStream = await httpClient.GetStreamAsync(imageUrl);

                        // Crop the image
                        using Image image = Image.Load(imageStream);
                        image.Mutate(x => x.Crop(new Rectangle(detectedFace.FaceRectangle.Left, detectedFace.FaceRectangle.Top, detectedFace.FaceRectangle.Width, detectedFace.FaceRectangle.Height)));
                        
                        // Save the cropped image to a memory stream
                        MemoryStream returnStream = new MemoryStream();
                        image.SaveAsJpeg(returnStream);
                        returnStream.Seek(0, SeekOrigin.Begin);
                        
                        OkObjectResult response = new OkObjectResult(returnStream);
                        response.ContentTypes.Add("image/jpeg");
                        return response;
                    }
                    else
                    {
                        return new BadRequestObjectResult("The image quality is too low for recognition");
                    }
                }
            }
            else
            {
                return new BadRequestObjectResult("Please pass a imageUrl on the query string");
            }

        }
    }
}
