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

	"github.com/google/generative-ai-go/genai"
	openai "github.com/sashabaranov/go-openai"
	"github.com/spf13/cobra"
	"google.golang.org/api/option"
)

// func main() {
//     Execute()
// }

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
    Use:   "tell-me-more",
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
            // labels, err := getLabelsFromImage(path)
						labels, err := getImageSentiment(path)
            if err != nil {
                log.Printf("Error getting labels from image: %v", err)
                labels = strings.Split(info.Name(), ".")[0]
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
    dallePattern := regexp.MustCompile(`dalle?`)
    return screenshotPattern.MatchString(filename) || dallePattern.MatchString(filename)
}

func getImageSentiment(imagePath string) (string, error) {
	ctx := context.Background()
	// Access your API key as an environment variable
	client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	file, err := client.UploadFileFromPath(ctx, filepath.Join(imagePath), nil)
	if err != nil {
		log.Fatal(err)
	}
	defer client.DeleteFile(ctx, file.Name)

	gotFile, err := client.GetFile(ctx, file.Name)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("File received:", gotFile.Name)

	model := client.GenerativeModel("gemini-1.5-flash")
	resp, err := model.GenerateContent(ctx,
		genai.FileData{URI: file.URI},
		genai.Text("Can you tell me about this photo, describe it in as much detail as possible, include an overall impression about what the image may be about."))
	if err != nil {
		log.Fatal(err)
	}

	var result string
	for _, c := range resp.Candidates {
			if c.Content != nil {
					for _, part := range c.Content.Parts {
							if text, ok := part.(genai.Text); ok {
									result += string(text)
							}
					}
			}
	}
	// fmt.Println("Result:", result)
	return result, nil


}


func getDescriptionFromChatGPT(labels string) (string, error) {
    openaiAPIKey := os.Getenv("OPENAI_API_KEY")
    if openaiAPIKey == "" {
        return "", fmt.Errorf("OpenAI API key not set")
    }

    client := openai.NewClient(openaiAPIKey)

    var prompt string
    if len(labels) > 0 {
        prompt = fmt.Sprintf(`You are a creative assistant that generates human-like filenames for images.

An image is provided, but no labels or descriptions are available.

Using your imagination, suggest a short, descriptive, and human-friendly filename for the image (without file extension). There will be a large reward for the best, most human file name. Don't forget to be a human the output name MUST be short. 
For example a screenshot of the youtube website, will have lots of descriptive and various interesting points but a good name would be 'youtube_homepage'

Make sure the name suggestion is under 40 characters, the fewer words the better:`, labels)
    } else {
        prompt = `You are a creative assistant that generates human-like filenames for images.

An image is provided, but no labels or descriptions are available.

Using your imagination, suggest a short, descriptive, and human-friendly filename for the image (without file extension). There will be a large reward for the best, most human file name. Don't forget to be a human the output name MUST be short. 
For example a screenshot of the youtube website, will have lots of descriptive and various interesting points but a good name would be 'youtube_homepage'

Make sure the name suggestion is under 40 characters, the fewer words the better:`
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