import {LitElement, html} from 'lit';

export class RangeInputElt extends LitElement {
    static properties = {
        label: {type: String},
        id: {type: String},
        min: {type: Number},
        max: {type: Number},
        step: {type: Number},
        value: {type: Number}
    };

    constructor() {
        super();
        this.min = 0;
        this.max = 1;
        this.step = 0.01;
        this.value = 0.5;
    }

    render() {
        return html`
            <div>
                <label for="${this.id}">${this.label}:</label>
                <input type="range"
                       id="${this.id}"
                       min="${this.min}"
                       max="${this.max}"
                       step="${this.step}"
                       value="${this.value}"
                       @input="${this._updateValue}">
                <span id="${this.id}Value">${this.value}</span>
            </div>
        `;
    }

    _updateValue(e) {
        this.value = e.target.value;
    }

    static defineCustomElt() {
        customElements.define('range-input', this);
    }
}


