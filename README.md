# NeoJD
## Introduction
Neonatal jaundice is a common condition that affects newborn infants, characterized by a yellow discoloration of the skin and 
whites of the eyes due to a build-up of bilirubin in the bloodstream. If left untreated, it can lead to serious health complications such as brain damage.

## WorkFlow
- This research commenced with the acquisition of a neonatal dataset, comprising of both visual imagery and corresponding bilirubin measurements.
- The subsequent step involved the extraction of the region of interest (ROI) from the image, specifically the skin area.
- Since, all the other part was noise, it was darkened.
- Now, the computation of the mean RGB values of the altered image was calculated. 
- Afterwards, the data was trained utilising the mean of the RGB value and bilirubin value as the primary features.
- In the testing phase, it has accuracy of almost 88%.

### Conceptual Image

![Conceptual Image](https://github.com/Sanchit71/Neonatal-Jaundice/assets/102990660/f29429e4-4415-46a2-b664-a7e50f3e0763)


## !IMPORTANT
- Due to privacy concern, the code and images of the neonates cannot be provided.
- For your reference, the presentation of the NeoJD has been provided.
