package cmd

import (
	"context"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	// "cloud.google.com/go/vision/v2"
	vision "cloud.google.com/go/vision/apiv1"
	visionpb "cloud.google.com/go/vision/v2/apiv1/visionpb"
	openai "github.com/sashabaranov/go-openai"
	"github.com/spf13/cobra"
)

// func main() {
//     Execute()
// }

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
    Use:   "screenshot-renamer",
    Short: "A CLI tool to rename image files based on their content",
    Run: func(cmd *cobra.Command, args []string) {
        if len(args) < 1 {
            fmt.Println("Please provide a directory to search")
            return
        }
        searchDirectory(args[0])
    },
}

func Execute() {
    if err := rootCmd.Execute(); err != nil {
        fmt.Println(err)
        os.Exit(1)
    }
}

func searchDirectory(dir string) {
    err := filepath.Walk(dir, func(path string, info fs.FileInfo, err error) error {
        if err != nil {
            return err
        }

        if !info.IsDir() && isTargetFile(info.Name()) {
            fmt.Printf("Found target file: %s\n", path)
            labels, err := getLabelsFromImage(path)
            if err != nil {
                log.Printf("Error getting labels from image: %v", err)
                labels = []string{}
            }

            description, err := getDescriptionFromChatGPT(labels)
            if err != nil {
                log.Printf("Error getting description from ChatGPT: %v", err)
                return nil
            }

            fmt.Printf("Suggested description: %s\n", description)
            fmt.Print("Do you want to rename the file? (y/n): ")
            var input string
            fmt.Scanln(&input)
            if strings.ToLower(input) == "y" {
                renameFile(path, description)
            }
        }
        return nil
    })

    if err != nil {
        log.Fatalf("Error walking the path %q: %v\n", dir, err)
    }
}

func isTargetFile(filename string) bool {
    filename = strings.ToLower(filename)
    screenshotPattern := regexp.MustCompile(`screenshot`)
    dallePattern := regexp.MustCompile(`dall[eé]?`)
    return screenshotPattern.MatchString(filename) || dallePattern.MatchString(filename)
}

func getLabelsFromImage(imagePath string) ([]string, error) {
    ctx := context.Background()

    client, err := vision.NewImageAnnotatorClient(ctx)
    if err != nil {
        return nil, fmt.Errorf("failed to create vision client: %v", err)
    }
    defer client.Close()

    // Open the image file
    file, err := os.Open(imagePath)
    if err != nil {
        return nil, fmt.Errorf("failed to open image file: %v", err)
    }
    defer file.Close()

    // Create the image using the helper function
    image, err := vision.NewImageFromReader(file)
    if err != nil {
        return nil, fmt.Errorf("failed to create image from file: %v", err)
    }

    // Create an AnnotateImageRequest with multiple features
    req := &visionpb.AnnotateImageRequest{
        Image: image,
        Features: []*visionpb.Feature{
            {Type: visionpb.Feature_LABEL_DETECTION},
            {Type: visionpb.Feature_WEB_DETECTION},
            {Type: visionpb.Feature_TEXT_DETECTION},
            {Type: visionpb.Feature_OBJECT_LOCALIZATION},
            {Type: visionpb.Feature_IMAGE_PROPERTIES},
        },
    }

    // Send the request
        // Send the request
    reqs := []*visionpb.AnnotateImageRequest{req}
    batchReq := &visionpb.BatchAnnotateImagesRequest{Requests: reqs}
    resp, err := client.BatchAnnotateImages(ctx, batchReq)
    if err != nil {
        return nil, fmt.Errorf("failed to annotate image: %v", err)
    }

    // Check for errors in response
    if len(resp.Responses) == 0 {
        return nil, fmt.Errorf("no response from vision API")
    }
    res := resp.Responses[0]
    if res.Error != nil {
        return nil, fmt.Errorf("vision API error: %v", res.Error.Message)
    }

    labels := []string{}

    // Process label annotations
    for _, annotation := range res.LabelAnnotations {
        labels = append(labels, annotation.Description)
    }

    // Process web detections
    if res.WebDetection != nil {
        web := res.WebDetection
        for _, entity := range web.WebEntities {
            if entity.Description != "" {
                labels = append(labels, entity.Description)
            }
        }
    }

    // Process text annotations
    if res.TextAnnotations != nil {
        for _, text := range res.TextAnnotations {
            labels = append(labels, text.Description)
        }
    }

    // Process localized object annotations
    if res.LocalizedObjectAnnotations != nil {
        for _, obj := range res.LocalizedObjectAnnotations {
            labels = append(labels, obj.Name)
        }
    }

    // Remove duplicates
    labels = removeDuplicates(labels)

    // Log labels for debugging
    // log.Printf("Labels for %s: %v", imagePath, labels)

    return labels, nil
}


func removeDuplicates(elements []string) []string {
    encountered := map[string]bool{}
    result := []string{}

    for _, v := range elements {
        if !encountered[v] {
            encountered[v] = true
            result = append(result, v)
        }
    }

    return result
}

func getDescriptionFromChatGPT(labels []string) (string, error) {
    openaiAPIKey := os.Getenv("OPENAI_API_KEY")
    if openaiAPIKey == "" {
        return "", fmt.Errorf("OpenAI API key not set")
    }

    client := openai.NewClient(openaiAPIKey)

    var prompt string
    if len(labels) > 0 {
        prompt = fmt.Sprintf(`You are a creative assistant that generates human-like filenames for images based on their content.

Given the following labels describing an image:
%s

Using these labels, write a short, descriptive, imaginative, and human-friendly filename for the image (without file extension). There will be a large reward for the best, most human file name:`, strings.Join(labels, ", "))
    } else {
        prompt = `You are a creative assistant that generates human-like filenames for images.

An image is provided, but no labels or descriptions are available.

Using your imagination, suggest a short, descriptive, and human-friendly filename for the image (without file extension). There will be a large reward for the best, most human file name. Don't forget to be a human the output name MUST be short. 
For example a screenshot of the youtube website, will have lots of descriptive and various interesting points but a good name would be 'youtube_homepage':`
    }

    ctx := context.Background()
    resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
        Model: openai.GPT4, // Use openai.GPT3Dot5Turbo if GPT-4 is not available
        Messages: []openai.ChatCompletionMessage{
            {
                Role:    openai.ChatMessageRoleUser,
                Content: prompt,
            },
        },
        MaxTokens:   100,
        Temperature: 0.9,
    })
    if err != nil {
        return "", fmt.Errorf("ChatGPT API error: %v", err)
    }

    if len(resp.Choices) > 0 {
        return strings.TrimSpace(resp.Choices[0].Message.Content), nil
    }

    return "", fmt.Errorf("no response from ChatGPT API")
}

func renameFile(path, description string) {
    dir := filepath.Dir(path)
    ext := filepath.Ext(path)
    newName := fmt.Sprintf("%s/%s%s", dir, sanitizeFileName(description), ext)

    err := os.Rename(path, newName)
    if err != nil {
        log.Fatalf("Failed to rename file: %v", err)
    }
    fmt.Printf("Renamed %s to %s\n", path, newName)
}

func sanitizeFileName(name string) string {
    reg := regexp.MustCompile(`[^\w\- ]`)
    cleanName := reg.ReplaceAllString(name, "")
    cleanName = strings.ReplaceAll(strings.TrimSpace(cleanName), " ", "_")
    return cleanName
}