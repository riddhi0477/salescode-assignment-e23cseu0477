I used Laplacian sharpness, Canny edge density, a uniform Local Binary Pattern texture histogram, specular highlight / glare fraction, HSV saturation stats, color-channel balance and more to differentiate between the real and through screen taken images . I resized the images to a max side of 800px. These ~33 features go into a LogisticRegression model.
Accuracy: 94.0% on 5-fold stratified cross-validation
Latency: ~245ms per image on CPU (laptop, single core, no GPU) 
Cost: $0 per image — runs fully on-device
What I'd improve with more time:I'd want to push the decision threshold above 0.5 and treat a middle band (~0.3–0.7) as send to a human rather than auto-deciding