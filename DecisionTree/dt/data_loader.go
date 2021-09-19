package dt

import (
	"encoding/csv"
	"fmt"
	"os"
)

func Load(filename string, trainScale int) ([][]string, [][]string, []string) {
	file, err := os.Open(filename)
	if err != nil {
		fmt.Errorf("Error when loading file " + err.Error())
		return nil, nil, nil
	}

	err = file.Close()
	if err != nil {
		fmt.Errorf("Error when closing file " + err.Error())
	}

	reader := csv.NewReader(file)

	var features []string
	var trainData [][]string
	var testData [][]string

	data, err := reader.Read()
	if err != nil {
		fmt.Errorf("Error when reader csv file " + err.Error())
	}

}
